import os
import torch
import logging
import numpy as np
import torch.distributed as dist
from typing import Dict
from trl import get_peft_config
from peft import get_peft_model
from ..config import CriticConfig
from rllava.data.protocol import DataProto
from rllava.utils.logger.aggregate_logger import print_rank_0
from rllava.utils.torch_dtypes import PrecisionType
from tqdm import tqdm
from codetiming import Timer
from collections import defaultdict
from transformers import AutoModelForTokenClassification
from rllava.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from rllava.utils.flops_counter import FlopsCounter
from ..utils.core_algos import compute_value_loss
from rllava.utils.py_functional import append_to_dict
from rllava.engine import EngineFactory
from rllava.utils.model_utils import print_model_size
from rllava.utils.performance import log_gpu_memory_usage
from rllava.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch



try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("RLLAVA_LOGGING_LEVEL", "WARN"))


class Critic():
    def __init__(self, config: CriticConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None

        self.accelerator = EngineFactory(config.strategy)(config)

    def initialize(self):
        self._init_model_and_optimizer(AutoModelForTokenClassification)
        self.flops_counter = FlopsCounter(self.config.model)

    def load_checkpoint(self, checkpoint_path: str):
        self.accelerator.load_state(checkpoint_path) 

    def save_checkpoint(self, checkpoint_path: str, save_model_only: bool = False):
        if self.accelerator.is_main_process:
            if save_model_only:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                torch.save(unwrapped_model.state_dict(), os.path.join(checkpoint_path, "model.pt"))
            else:
                self.accelerator.save_state(checkpoint_path)
        self.accelerator.wait_for_everyone()

    def _init_model_and_optimizer(self, model_class):
        """Initialize model and optimizer directly in Actor class."""
        print_rank_0("Initializing model and optimizer in Actor class...")
        
        # Load model
        torch_dtype = self.config.model.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)
        
        init_weight = self.accelerator.get_init_weight_context(use_meta_tensor=not self.config.model.tie_word_embeddings)
        with init_weight():
            self.model = model_class.from_pretrained(
                self.config.model.model_path,
                config=self.config.model,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=self.config.model.trust_remote_code,
            )
        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        peft_config = get_peft_config(self.config.model)
        
        if peft_config is not None:
            self.model.enable_input_require_grads()
            # If PEFT is used, wrap the model with PEFT
            peft_model = get_peft_model(self.model, peft_config)
            self.model = peft_model

        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print_model_size(self.model)
        log_gpu_memory_usage(f"After init Actor from HF AutoModel", logger=logger)
        self.model = self.accelerator.prepare(self.model)
        log_gpu_memory_usage(f"After Actor Accelerator prepare", logger=logger)


    def _forward_micro_batch(self, micro_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
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

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.model(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating
            values_rmpad = output.logits
            values_rmpad = values_rmpad.squeeze(0)  # (total_nnz)

            # gather output if sp > 1
            if self.config.ulysses_size > 1:
                values_rmpad = gather_outputs_and_unpad(values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad it back
            values = pad_input(values_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            values = values[:, -response_length - 1 : -1]
        else:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            values: torch.Tensor = output.logits
            values = values[:, -response_length - 1 : -1].squeeze(-1)  # (bsz, response_length, vocab_size)

        return values

    @torch.no_grad()
    def compute_values(self, data: DataProto):
        self.model.eval()

        select_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = self.config.log_prob_micro_batch_size_per_gpu * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.log_prob_micro_batch_size_per_gpu)

        values_lst = []
        if self.accelerator.is_main_process:
            micro_batches = tqdm(micro_batches, desc="Compute values", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            values = self._forward_micro_batch(model_inputs)
            values_lst.append(values)

        values = torch.concat(values_lst, dim=0)

        if self.config.dynamic_batching:
            values = restore_dynamic_batch(values, batch_idx_list)

        values = values * data.batch["response_mask"]  

        return DataProto.from_dict(tensors={"values": values})
    
    def _compute_values(self, module, data: DataProto):
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.log_prob_micro_batch_size_per_gpu
        )
        values_lst = []
        if self.worker.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute values", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            values = self._forward_micro_batch(module, model_inputs)
            values_lst.append(values)

        values = torch.concat(values_lst, dim=0)
        responses = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        response_length = responses.size(1)
        values = values * attention_mask[:, -response_length:]
        return values
    
    def _update(self, data: DataProto):
        self.model.train()

        select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "values", "returns"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.accelerator.is_main_process:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                if self.config.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = self.config.ppo_micro_batch_size_per_gpu * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                if self.accelerator_manager.accelerator.is_main_process:
                    micro_batches = tqdm(micro_batches, desc="Update critic", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    values = model_inputs["values"]
                    returns = model_inputs["returns"]

                    vpreds = self._forward_micro_batch(model_inputs)
                    vf_loss, vf_metrics = compute_value_loss(
                        vpreds=vpreds,
                        returns=returns,
                        values=values,
                        response_mask=response_mask,
                        cliprange_value=self.config.cliprange_value,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )
                    loss = vf_loss * torch.sum(response_mask) * self.accelerator_manager.accelerator.num_processes / total_response_tokens
                    loss.backward()

                    batch_metrics = {
                        "critic/vf_loss": vf_loss.detach().item(),
                        "critic/vf_clipfrac": vf_metrics["vf_clipfrac"],
                        "critic/vpred_mean": vf_metrics["vpred_mean"],
                    }
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"critic/grad_norm": grad_norm.detach().item()})

        return metrics
    
    def update(self, data: DataProto) -> DataProto:
        """Update critic network.
        
        This method contains the algorithm logic for critic updates.
        Distributed operations are handled by the worker.
        
        Args:
            batch: Training batch data
            
        Returns:
            Training metrics
        """
        # Execute with distributed operations handled by worker
        with Timer(name="update_critic", logger=None) as timer:
            metrics = self._update(data)

        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics["perf/mfu_critic"] = (
            estimated_flops * self.config.ppo_epochs / (promised_flops * self.worker.world_size)
        )
        
        # Add learning rate and other metrics
        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]
        metrics["critic/lr"] = lr
        
        # Wrap metrics in DataProto
        return DataProto(
            non_tensor_batch={
                key: np.array([value] if np.isscalar(value) else value) for key, value in metrics.items()
            }
        )