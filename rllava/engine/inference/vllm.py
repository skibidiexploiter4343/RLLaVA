import inspect
import torch
from typing import Optional, List, Iterable, Tuple, TYPE_CHECKING, Dict
from transformers import PreTrainedTokenizer, ProcessorMixin
from contextlib import contextmanager
from .base import InferenceEngine
from .. import register_engine
from rllava.utils import torch_functional as VF
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from rllava.utils.torch_dtypes import PrecisionType
from rllava.data.protocol import DataProto
from vllm import RequestOutput
from vllm.lora.request import LoRARequest
from peft.utils.save_and_load import get_peft_model_state_dict
import time
from .utils import TensorLoRARequest, VLLMHijack
from dataclasses import asdict
from .base import _get_logit_bias, _process_multi_modal_data, _repeat_interleave
from rllava.utils.model_utils import print_gpu_memory_usage
if TYPE_CHECKING:
    from rllava.ppo.config import RolloutConfig



@register_engine("vllm")
class VLLMEngine(InferenceEngine):
    def __init__(
        self,
        model_name_or_path: str,
        config: 'RolloutConfig',
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        super().__init__(model_name_or_path, config, tokenizer, processor)

        if config.vllm.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")
        self.lora_kwargs = (
            {"enable_lora": True, "max_loras": 1, "max_lora_rank": 8}
            if config.load_format == "safetensors"
            else {}
        )
        VLLMHijack.hijack()
        self.inference_engine = LLM(
            model=model_name_or_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format=config.load_format,
            dtype=config.vllm.dtype,
            seed=config.seed,
            max_model_len=config.vllm.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.vllm.gpu_memory_utilization,
            max_num_batched_tokens=config.vllm.max_num_batched_tokens,
            disable_log_stats=config.vllm.disable_log_stats,
            enforce_eager=config.vllm.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.vllm.enable_chunked_prefill,
            enable_sleep_mode=True,
            **self.lora_kwargs,
            **self.engine_kwargs,
        )

        # Offload vLLM model to reduce peak memory usage at init
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)
        self.loaded = False

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def generate(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("VLLM input preprocessing size mismatch across TP ranks.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": _process_multi_modal_data(
                            multi_modal_data,
                            prompts.meta_info["min_pixels"],
                            prompts.meta_info["max_pixels"],
                            prompts.meta_info["video_fps"],
                        ),
                    }
                )
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]
        
        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/lora-path")
                ] * batch_size
        
        with self.update_sampling_params(**prompts.meta_info):
            completions: List[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm, lora_request=lora_requests,
            )
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)

    def update_weights(self, model):
        if self.lora_kwargs:
            weights = get_peft_model_state_dict(model)
            peft_config = model.peft_config.get("default", None)
            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_request = TensorLoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="lora_path",
                peft_config=asdict(peft_config),
                lora_tensors=weights,
            )
            self.inference_engine.llm_engine.add_lora(lora_request)
        else:
            weights = model.state_dict()
            weights_iter = self._make_weight_iterator(weights)
            vllm_model = (
                self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            )
            vllm_model.load_weights(weights_iter)
            torch.cuda.empty_cache()
            print_gpu_memory_usage("After sync model weights in vllm engine")

    def load(self, model):
        # Prepare vLLM engine and sync model weights per rollout; always wake then sleep
        torch.cuda.empty_cache()
        assert self.loaded is False, "vllm engine has already been loaded"
        self.loaded = True
        
        print_gpu_memory_usage("Before vllm wake up in vllm engine")
        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=["weights"])
        else:
            self.inference_engine.wake_up()
        
        self.update_weights(model)
        
        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=["kv_cache"])

        print_gpu_memory_usage("After vllm wake up in vllm engine")

    def offload(self):
        assert self.loaded is True, "vllm engine has not been loaded"
        
        print_gpu_memory_usage("Before vllm offload in vllm engine")
        self.inference_engine.sleep(level=1)
        print_gpu_memory_usage("After vllm offload in vllm engine")

        torch.cuda.empty_cache()
        self.loaded = False

    def _make_weight_iterator(self, weights: Dict[str, torch.Tensor]):
        """Create an iterator over model weights.
        
        Args:
            actor_weights: Model weights dictionary
            
        Returns:
            Iterator over (name, tensor) pairs
        """
        for name in sorted(weights.keys()):
            tensor = weights[name]
            # Handle DTensor for distributed training
            if hasattr(tensor, 'full_tensor'):
                yield name, tensor.full_tensor() if self.world_size != 1 else tensor
            else:
                yield name, tensor 

    