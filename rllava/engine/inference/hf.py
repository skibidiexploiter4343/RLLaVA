import os
import torch
import logging
import contextlib
from tqdm import tqdm
from typing import Optional, TYPE_CHECKING
from transformers import PreTrainedTokenizer, ProcessorMixin
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from .base import InferenceEngine
from .. import register_engine
from rllava.utils import torch_functional as VF
from tensordict import TensorDict
from transformers import GenerationConfig
from rllava.data.protocol import DataProto
from rllava.utils.performance import log_gpu_memory_usage



if TYPE_CHECKING:
    from rllava.ppo.config import RolloutConfig


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("RLLAVA_LOGGING_LEVEL", "WARN"))


@register_engine("hf")
class HFEngine(InferenceEngine):
    def __init__(
        self,
        model_name_or_path: str,
        config: 'RolloutConfig',
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        super().__init__(model_name_or_path, config, tokenizer, processor)
        self.model = None
        self.loaded = False

    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", getattr(self.config, "top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", getattr(self.config, "top_k", 0)))  

        input_ids = prompts.batch["input_ids"]  # (bs, prompt_length)
        prompt_length = input_ids.size(1)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]
        
        # When temperature is 0.0, use greedy decoding (do_sample=False)
        # Otherwise use sampling (do_sample=True)
        do_sample = temperature > 0.0
        
        self.generation_config = GenerationConfig(
            max_new_tokens=response_length,
            do_sample=do_sample,
            num_beams=1,
            top_p=top_p if do_sample else None,
            top_k=top_k if do_sample else None,
            temperature=temperature if do_sample else None, 
            num_return_sequences=1,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        # Prepare generation arguments based on task type
        prompt_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "output_scores": False,  # this is potentially very large
            "return_dict_in_generate": True,
            "use_cache": True,
        }
        # Check if this is a multimodal task by checking for image-related fields
        is_multimodal = "pixel_values" in prompts.batch and "image_grid_thw" in prompts.batch
        if is_multimodal:
            prompt_inputs["pixel_values"] = prompts.batch["pixel_values"]
            prompt_inputs["image_grid_thw"] = prompts.batch["image_grid_thw"]    

        self.model.eval()
        param_ctx = contextlib.nullcontext()
        if isinstance(self.model, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.model, writeback=False, recurse=False)

        with param_ctx, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.model.generate(**prompt_inputs, 
                                         generation_config=self.generation_config)
        seq = output.sequences
        generated_batch_size = seq.size(0)  # bs * num_return_sequences

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(size=(generated_batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)
        assert seq.shape[1] == sequence_length

        prompt = seq[:, :prompt_length]  # (generated_batch_size, prompt_length)
        response = seq[:, prompt_length:]  # (generated_batch_size, response_length)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)
        
        # Handle Qwen2-VL MRoPE 3D position_ids
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(generated_batch_size, 1, -1).expand(generated_batch_size, 3, -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = VF.get_response_mask(
            response_ids=response, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "response_mask": response_attention_mask,
                "position_ids": position_ids,
            },
            batch_size=generated_batch_size,
        )
        # empty cache before compute old_log_prob
        torch.cuda.empty_cache()
        self.model.train()

        return DataProto(batch=batch)

    def generate(self, prompts: DataProto) -> DataProto:
        prompts = prompts.to(torch.cuda.current_device())
        num_return_sequences = prompts.meta_info.get("n", getattr(self.config, "n", 1))
        prompts = prompts.repeat(repeat_times=num_return_sequences, interleave=True)
        
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // getattr(self.config, "micro_batch_size", batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        
        # Add progress bar for batch processing
        batch_prompts_iter = tqdm(
            batch_prompts, 
            desc="Generating batches", 
            disable=False,
            total=len(batch_prompts)
        )
        
        output = [self._generate_minibatch(p) for p in batch_prompts_iter]
        output = DataProto.concat(output)
        return output

    def update_weights(self, model):
        self.model = model

    def load(self, model):
        torch.cuda.empty_cache()
        assert self.loaded is False, "hf engine has already been loaded"

        log_gpu_memory_usage("Before hf wake up in hf engine", logger=logger)  
        self.update_weights(model)
        log_gpu_memory_usage("After hf wake up in hf engine", logger=logger)
        self.loaded = True

    def offload(self):
        assert self.loaded is True, "hf engine has not been loaded"
        self.model = None
        self.loaded = False
        

    