import torch
import psutil
import numpy as np
import logging
import os

from typing import Dict
import torch.nn.functional as F
from trl import get_peft_config
from peft import get_peft_model
from deprecated import deprecated
from ..config import ActorConfig
from rllava.data.protocol import DataProto
from rllava.utils.device import get_torch_device
from rllava.utils.logger.aggregate_logger import print_rank_0
from rllava.utils.torch_dtypes import PrecisionType
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForVision2Seq, AutoModelForCausalLM, AutoConfig
from einops import rearrange
from rllava.utils import torch_functional as VF
from rllava.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from rllava.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from rllava.utils.flops_counter import FlopsCounter
from rllava.utils.py_functional import append_to_dict
from codetiming import Timer
from ..utils.core_algos import kl_penalty, agg_loss
from rllava.utils.model_utils import print_model_size
from rllava.utils.performance import log_gpu_memory_usage
from rllava.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from rllava.utils.device import get_device_name
from rllava.engine import EngineFactory
from rllava.utils.train_utils import find_all_linear_names
from rllava.utils.dist_utils import is_rank0
from contextlib import nullcontext

try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass



logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("RLLAVA_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class Actor():
    def __init__(self, config: ActorConfig, policy_loss=None, tokenizer=None, processor=None):
        self.config = config
        self.policy_loss = policy_loss
        self.tokenizer = tokenizer
        self.processor = processor        
        self.model = None
        self.is_peft_model = False
        self.optimizer = None
        self.lr_scheduler = None

        self.accelerator = EngineFactory(config.strategy)(config)
            
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

        world_size = self.accelerator.num_processes
        self.config.global_batch_size_per_device = self.config.ppo_mini_batch_size // (world_size // self.config.ulysses_size)
        if self.config.global_batch_size_per_device == 0:
            raise ValueError(f"Actor global batch size * ulysses size must be larger than num gpus.")

        if self.config.global_batch_size_per_device % self.config.ppo_micro_batch_size_per_gpu != 0:
            raise ValueError(f"Actor global batch size per device must be divisible by the micro batch size.")
        
        print_rank_0(f"Actor will use global batch size per device {self.config.global_batch_size_per_device}.")

    def initialize(self, model_config: AutoConfig):
        if type(model_config) in AutoModelForVision2Seq._model_mapping.keys():
            model_class = AutoModelForVision2Seq
        else:
            model_class = AutoModelForCausalLM
        
        # Initialize model directly in Actor class
        self.init_model(model_class, model_config)
        self.init_optimizer()
        self.flops_counter = FlopsCounter(model_config)

    def unwrap_model(self):
        return self.accelerator.unwrap_model(self.model)
    
    def unwrap_model_for_generation(self):
        return self.accelerator.unwrap_model_for_generation(self.model, self.is_peft_model)

    def load_checkpoint(self, checkpoint_path: str):
        self.accelerator.load_state(self.model, self.optimizer, self.lr_scheduler, checkpoint_path) 

    def save_checkpoint(self, checkpoint_path: str, save_model_only: bool = False):
        if save_model_only:
            if self.accelerator.is_main_process:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                torch.save(unwrapped_model.state_dict(), os.path.join(checkpoint_path, "model.pt"))
        else:
            # Call save_state on ALL ranks; Accelerate will coordinate and only write once.
            self.accelerator.save_state(self.model, self.optimizer, self.lr_scheduler, checkpoint_path)
        self.accelerator.wait_for_everyone()

    def init_model(self, model_class, model_config):
        """Initialize model in Actor class."""
        log_gpu_memory_usage(f"Before init Actor from HF AutoModel", logger=logger)
        # Load model
        torch_dtype = self.config.model.torch_dtype
        if torch_dtype is None:
            # torch_dtype = torch.float32
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)
        
        init_weight = self.accelerator.get_init_weight_context(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.accelerator.device_mesh)
        with init_weight():
            self.model = model_class.from_pretrained(
                self.config.model.model_path,
                config=model_config,
                torch_dtype=torch_dtype,
                attn_implementation=self.config.model.attn_implementation,
                trust_remote_code=self.config.model.trust_remote_code,
            )
            self.model.to(torch_dtype)
        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        self.config.model.lora_target_modules = find_all_linear_names(self.model, ['visual','connector', 'vision_tower'] )
        peft_config = get_peft_config(self.config.model)
        
        if peft_config is not None:
            self.is_peft_model = True
            self.model.enable_input_require_grads()
            # If PEFT is used, wrap the model with PEFT
            peft_model = get_peft_model(self.model, peft_config)
            self.model = peft_model

        if is_rank0(): 
            print_model_size(self.model)
        log_gpu_memory_usage(f"After init Actor from HF AutoModel", logger=logger)
        self.model = self.accelerator.prepare(self.model)
        log_gpu_memory_usage(f"After Actor Accelerator prepare", logger=logger)


        # Handle reference model if needed
        if self.config.use_kl_loss or self.config.use_kl_in_reward:
            if peft_config is not None:
                # If PEFT is used, disable adapters for reference
                ref_model = self.model.disable_adapter()
                self.ref_model = ref_model
            else:
                # Load separate reference model
                with init_weight():
                    self.ref_model = model_class.from_pretrained(
                        self.config.model.model_path,
                        config=self.config.model,
                        torch_dtype=torch.bfloat16,
                        attn_implementation=self.config.model.attn_implementation,
                        trust_remote_code=self.config.model.trust_remote_code,
                    )
                for param in self.ref_model.parameters():
                    param.requires_grad = False

            log_gpu_memory_usage(f"After init Ref from HF AutoModel", logger=logger)
            self.ref_model = self.accelerator.prepare(self.ref_model, forward_only=True)
            log_gpu_memory_usage(f"After Ref Accelerator prepare", logger=logger)
               
    def init_optimizer(self):
        # Create optimizer
        if self.config.optim.strategy == "adamw":
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.optim.lr,
                betas=self.config.optim.betas,
                weight_decay=self.config.optim.weight_decay,
                fused=True,
            )
        elif self.config.optim.strategy == "adamw_bf16":
            from utils.torch_functional import AnyPrecisionAdamW
            self.optimizer = AnyPrecisionAdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.optim.lr,
                betas=self.config.optim.betas,
                weight_decay=self.config.optim.weight_decay,
            )
        else:
            raise NotImplementedError(f"Optimizer {self.config.optim.strategy} not supported.")
        
        # Create learning rate scheduler   
        if self.config.optim.lr_warmup_steps is not None:
            num_warmup_steps = self.config.optim.lr_warmup_steps
        else:
            num_warmup_steps = int(self.config.optim.lr_warmup_ratio * self.config.optim.training_steps)
        
        if self.config.optim.lr_scheduler_type == "constant":
            self.lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps
            )
        elif self.config.optim.lr_scheduler_type == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.config.optim.training_steps,
                min_lr_ratio=self.config.optim.min_lr_ratio,
                num_cycles=self.config.optim.num_cycles,
            )

        self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.optimizer, self.lr_scheduler)
        log_gpu_memory_usage(f"After Optimizer and LR Scheduler Accelerator prepare", logger=logger)

    @deprecated("Use init_model and init_optimizer instead")
    def _init_model_and_optimizer(self, model_class, model_config):
        """Initialize model and optimizer directly in Actor class."""
        print_rank_0("Initializing model and optimizer in Actor class...")
        
        # Load model
        torch_dtype = self.config.model.torch_dtype
        if torch_dtype is None:
            torch_dtype = torch.float32
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)
        
        init_weight = self.accelerator.get_init_weight_context(use_meta_tensor=not model_config.tie_word_embeddings)
        with init_weight():
            self.model = model_class.from_pretrained(
                self.config.model.model_path,
                config=model_config,
                torch_dtype=torch_dtype,
                attn_implementation=self.config.model.attn_implementation,
                trust_remote_code=self.config.model.trust_remote_code,
            )
        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.config.model.lora_target_modules = find_all_linear_names(self.model, ['visual','connector', 'vision_tower'] )
        peft_config = get_peft_config(self.config.model)
        
        if peft_config is not None:
            self.is_peft_model = True
            self.model.enable_input_require_grads()
            # If PEFT is used, wrap the model with PEFT
            peft_model = get_peft_model(self.model, peft_config)
            self.model = peft_model

        if is_rank0() == 0: 
            print_model_size(self.model)
        log_gpu_memory_usage(f"After init Actor from HF AutoModel", logger=logger)
        self.model = self.accelerator.prepare(self.model)
        log_gpu_memory_usage(f"After Actor Accelerator prepare", logger=logger)

        # Handle reference model if needed
        if self.config.use_kl_loss or self.config.use_kl_in_reward:
            if peft_config is not None:
                # If PEFT is used, disable adapters for reference
                ref_model = self.model.disable_adapter()
                self.ref_model = ref_model
            else:
                # Load separate reference model
                with init_weight():
                    self.ref_model = model_class.from_pretrained(
                        self.config.model.model_path,
                        config=self.config.model,
                        torch_dtype=torch.bfloat16,
                        attn_implementation=self.config.model.attn_implementation,
                        trust_remote_code=self.config.model.trust_remote_code,
                    )
                for param in self.ref_model.parameters():
                    param.requires_grad = False

            log_gpu_memory_usage(f"After init Ref from HF AutoModel", logger=logger)
            self.ref_model = self.accelerator.prepare(self.ref_model)
            log_gpu_memory_usage(f"After Ref Accelerator prepare", logger=logger)
               
        # Create optimizer
        if self.config.optim.strategy == "adamw":
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.optim.lr,
                betas=self.config.optim.betas,
                weight_decay=self.config.optim.weight_decay,
                fused=True,
            )
        elif self.config.optim.strategy == "adamw_bf16":
            from utils.torch_functional import AnyPrecisionAdamW
            self.optimizer = AnyPrecisionAdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.optim.lr,
                betas=self.config.optim.betas,
                weight_decay=self.config.optim.weight_decay,
            )
        else:
            raise NotImplementedError(f"Optimizer {self.config.optim.strategy} not supported.")
        
        # Create learning rate scheduler   
        if self.config.optim.lr_warmup_steps is not None:
            num_warmup_steps = self.config.optim.lr_warmup_steps
        else:
            num_warmup_steps = int(self.config.optim.lr_warmup_ratio * self.config.optim.training_steps)
        
        self.lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=self.optimizer, num_warmup_steps=num_warmup_steps
        )
        self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.optimizer, self.lr_scheduler)
        log_gpu_memory_usage(f"After Optimizer and LR Scheduler Accelerator prepare", logger=logger)

    def _forward_micro_batch(self, model, micro_batch: Dict[str, torch.Tensor], temperature: float, calculate_entropy: bool = False, return_logits: bool = False):
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            for input_dict in micro_batch["multi_modal_inputs"]:
                for key, value in input_dict.items():
                    multi_modal_inputs[key].append(value)

            for key, value in multi_modal_inputs.items():
                if len(value) != 0:
                    multi_modal_inputs[key] = torch.cat(value, dim=0)
                else:
                    multi_modal_inputs[key] = None

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = model(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            logits_slice = None
            if return_logits:
                if self.config.ulysses_size > 1:
                    logits_rmpad = gather_outputs_and_unpad(logits_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                full_logits = pad_input(
                    hidden_states=logits_rmpad, indices=indices, batch=batch_size, seqlen=seqlen
                )  # (bsz, seqlen, vocab)
                logits_slice = full_logits[:, -response_length - 1 : -1, :]
        else:
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)
            logits_slice = logits if return_logits else None

        # Calculate entropy if requested
        entropy = None
        if calculate_entropy:
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # (bsz, response_length)
        
        return entropy, log_probs, logits_slice

    @torch.no_grad()
    def compute_log_probs(self, data: DataProto, temperature: float = None, is_ref: bool = False) -> DataProto:
        """Compute log probabilities for the given batch.
        
        Args:
            batch: Input batch for log probability computation
            is_ref: Whether this is for reference model computation
            
        Returns:
            Log probabilities
        """
        data = data.to(torch.cuda.current_device())
        data.meta_info["temperature"] = temperature
        # Use Actor's own module for computation
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        micro_batches = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = self.config.log_prob_micro_batch_size_per_gpu * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.log_prob_micro_batch_size_per_gpu)

        log_probs_lst = []
        if self.accelerator.is_main_process:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        if is_ref:
            if self.is_peft_model == True:
                model = self.model
            else:
                model = self.ref_model
        else:
            model = self.model
        ctx = model.disable_adapter() if self.is_peft_model == True and is_ref else nullcontext()
        with ctx, self.accelerator.eval(model):
            for micro_batch in micro_batches:
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                _, log_probs, _ = self._forward_micro_batch(model=model, micro_batch=model_inputs, temperature=temperature)
                log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
        return log_probs
    
    def update(self, data: DataProto):
        # Perform training with Accelerator
        with Timer(name="update_policy", logger=None) as timer:
            with self.accelerator.train(model=self.model, optimizer=self.optimizer):
                metrics = self.update_policy(data=data)

        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        
        # Use accelerator's world size
        world_size = self.accelerator.num_processes
        metrics["perf/mfu/actor"] = estimated_flops * self.config.ppo_epochs / promised_flops / world_size
        metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

        lr = self.lr_scheduler.get_last_lr()[0]
        metrics["actor/lr"] = lr
        self.lr_scheduler.step()

        output = DataProto(
               non_tensor_batch={
                   key: np.array([value] if np.isscalar(value) else value) for key, value in metrics.items()
               }
           )

        return output
    
    def update_policy(self, data: DataProto):
        """Update policy using Accelerator for unified training management."""

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_probs")
        if self.config.tis_imp_ratio_cap > 0:
            assert "rollout_log_probs" in data.batch.keys(), (
                "Truncated Importance Sampling (TIS) requires to configure "
                "`actor_rollout_ref.rollout.calculate_log_probs=True` "
                "and is not currently supported in Server mode (agent loop)."
            )
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        # Calculate number of mini_batches to determine on_policy before actual data selection
        batch_size = len(data)
        num_mini_batches = (batch_size + self.config.global_batch_size_per_device - 1) // self.config.global_batch_size_per_device
        on_policy = num_mini_batches == 1 and self.config.ppo_epochs == 1

        if not on_policy:
            select_keys.append("old_log_probs")

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1, disable=not self.accelerator.is_main_process)

            for mini_batch in mini_batches:
                if self.config.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = self.config.ppo_micro_batch_size_per_gpu * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.global_batch_size_per_device // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                micro_batches = tqdm(micro_batches, desc="Update policy", position=2, disable=not self.accelerator.is_main_process)
                for micro_batch in micro_batches:
                    micro_batch_metrics = defaultdict(list)
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    rollout_log_probs = model_inputs["rollout_log_probs"] if self.config.tis_imp_ratio_cap > 0 else None
                    advantages = model_inputs["advantages"]

                    if self.config.dynamic_batching:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_micro_batch_size_per_gpu
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if self.config.entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_probs, logits_slice = self._forward_micro_batch(
                        model=self.model,
                        micro_batch=model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        return_logits=True if self.config.sft_loss_coef > 0 else False,
                    )

                    if on_policy:
                        old_log_probs = log_probs.detach()
                    else:
                        old_log_probs = model_inputs["old_log_probs"]

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = self.policy_loss(
                        old_log_prob=old_log_probs,
                        log_prob=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=self.config.loss_agg_mode,
                        config=self.config,
                        rollout_log_probs=rollout_log_probs,
                    )

                    if self.config.entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)
                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * self.config.entropy_coeff
                    else:
                        policy_loss = pg_loss
                    # UFT-style joint SFT loss against ground_truth (optional)
                    if getattr(self.config, "sft_loss_coef", 0.0) and self.config.sft_loss_coef > 0 and self.tokenizer is not None:
                        if "ground_truth" in model_inputs:
                            gt_texts = [str(x) for x in model_inputs["ground_truth"]]
                            tok = self.tokenizer(gt_texts, add_special_tokens=False, return_tensors="pt", padding=True, truncation=True)
                            target_ids = tok["input_ids"].to(logits_slice.device)
                            bsz, resp_len = logits_slice.size(0), logits_slice.size(1)
                            if target_ids.size(1) > resp_len:
                                target_ids = target_ids[:, :resp_len]
                            elif target_ids.size(1) < resp_len:
                                pad = torch.full((target_ids.size(0), resp_len - target_ids.size(1)), -100, dtype=target_ids.dtype, device=target_ids.device)
                                target_ids = torch.cat([target_ids, pad], dim=1)
                            ce = F.cross_entropy(logits_slice.reshape(-1, logits_slice.size(-1)), target_ids.reshape(-1), ignore_index=-100, reduction='none').reshape(bsz, resp_len)
                            sft_loss = agg_loss(loss_mat=ce, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)
                            policy_loss = policy_loss + self.config.sft_loss_coef * sft_loss
                            micro_batch_metrics["actor/sft_loss"] = sft_loss.detach().item() * loss_scale_factor
                    
                    if self.config.use_kl_loss:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_probs,
                            ref_logprob=ref_log_probs,
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef
     
                    loss = policy_loss * loss_scale_factor
                    self.accelerator.backward(loss)

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac_higher": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.optimizer.zero_grad()
        return metrics
    
    def _optimizer_step(self):
        """Perform optimizer step with Accelerator support."""
        grad_norm = self.accelerator.clip_grad_norm_(
                        self.model, self.config.max_grad_norm
                    )
            
        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        return grad_norm




