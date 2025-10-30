import torch
import numpy as np
import logging
import os

from typing import Dict
from ..config import ActorConfig
from rllava.data.protocol import DataProto
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForVision2Seq, AutoModelForCausalLM, AutoConfig
from einops import rearrange
from .actor import Actor
from rllava.model.patch.monkey_patch import apply_monkey_patch
from rllava.utils import torch_functional as VF
from rllava.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
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


class Ref():
    def __init__(self, config: ActorConfig, tokenizer=None, processor=None):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor        
        self.model = None

        self.accelerator = EngineFactory(config.strategy)(config)
            
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    def initialize(self, actor: Actor):
        self.actor = actor

        if not actor.is_peft_model:
            if type(actor.model_config) in AutoModelForVision2Seq._model_mapping.keys():
                model_class = AutoModelForVision2Seq
            else:
                model_class = AutoModelForCausalLM
            
            self.init_model(model_class, actor.model_config)

    def init_model(self, model_class, model_config):
        """Initialize model in Ref class."""
        log_gpu_memory_usage(f"Before init Ref from HF AutoModel", logger=logger)
        
        init_weight = self.accelerator.get_init_weight_context(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.accelerator.device_mesh)
        with init_weight():
            self.model = model_class.from_pretrained(
                self.config.model.model_path,
                config=model_config,
                torch_dtype=torch.bfloat16,
                attn_implementation=self.config.model.attn_implementation,
                trust_remote_code=self.config.model.trust_remote_code,
            )

            apply_monkey_patch(
                model=self.model,
                use_remove_padding=self.config.padding_free,
                ulysses_sp_size=self.config.ulysses_size
            )

            self.model.to(torch.bfloat16)
        
        for param in self.model.parameters():
            param.requires_grad = False

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if is_rank0(): 
            print_model_size(self.model)
        log_gpu_memory_usage(f"After init Ref from HF AutoModel", logger=logger)
        self.model = self.accelerator.prepare(self.model)
        log_gpu_memory_usage(f"After Ref Accelerator prepare", logger=logger)

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
            position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

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
                )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
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
    def compute_log_probs(self, data: DataProto, temperature: float = None) -> DataProto:
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

        if self.actor.is_peft_model:
            ctx = self.actor.model.disable_adapter()
            model = self.actor.model
        else:
            ctx = nullcontext()
            model = self.model
        with ctx, self.accelerator.eval(model):
            for micro_batch in micro_batches:
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                _, log_probs, _ = self._forward_micro_batch(model=model, micro_batch=model_inputs, temperature=temperature)
                log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
        return log_probs
