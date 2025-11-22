import torch
import numpy as np
from ..config import RolloutConfig
from rllava.engine import EngineFactory
from rllava.data.protocol import DataProto
from rllava.utils.model_utils import print_gpu_memory_usage
from typing import Callable
from transformers import GenerationConfig
from copy import deepcopy



__all__ = ["Rollout"]


class Rollout():
    def __init__(self, config: RolloutConfig, reward, tokenizer, processor, workflow=None):
        self.config = config
        self.n = config.n
        self.tokenizer = tokenizer
        self.processor = processor
        self.reward = reward
        self.workflow = workflow

    def initialize(self, model_path):
        self.model_path = model_path
        self.generation_config = GenerationConfig.from_pretrained(model_path)

        """Initialize the rollout engine for sequence generation."""
        print_gpu_memory_usage(f"Before {self.config.name} rollout engine init")
        engine_class = EngineFactory(self.config.name)
        self.rollout_engine = engine_class(
            model_name_or_path=self.model_path,
            config=self.config,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        print_gpu_memory_usage(f"After {self.config.name} rollout engine init")

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences using the rollout engine.
        
        Args:
            prompts: Input prompts for generation
            
        Returns:
            Generated sequences
        """
        if self.rollout_engine is None:
            raise RuntimeError("Rollout engine not initialized. Call init_model first.")
        
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        output = self.rollout_engine.generate(prompts=prompts)
        output = output.to("cpu")
        return output

    def generate_one_batch(self, data: DataProto, filter: Callable = lambda sample: sample, val=False) -> DataProto:  
        # uid
        import uuid
        data.non_tensor_batch["uid"] = np.array([
            str(uuid.uuid4()) for _ in range(len(data.batch))
        ], dtype=object)

        # pop keys for generation
        gen_batch = data.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
        )
        if val:
            gen_batch.meta_info = self.config.val_override_config
        else:
            gen_batch.meta_info = dict()

        gen_batch.meta_info["min_pixels"] = self.config.min_pixels
        gen_batch.meta_info["max_pixels"] = self.config.max_pixels
        gen_batch.meta_info["video_fps"] = self.config.video_fps
        if self.workflow is None:
        # DP handled by Accelerate's sharded dataloaders; generate normally on each rank's batch
            gen_batch_output = self.generate_sequences(gen_batch)
        else:
            gen_batch_output = self.workflow.generate(gen_batch)

        if val:
            data = data.repeat(repeat_times=self.config.val_override_config.get("n", 1), interleave=True)
            new_batch = data.union(gen_batch_output)
            return new_batch
        
        if self.config.adv_estimator == "remax":
            gen_baseline_batch = deepcopy(gen_batch)
            gen_baseline_batch.meta_info["temperature"] = 0
            gen_baseline_batch.meta_info["n"] = 1
            gen_baseline_output = self.generate_sequences(gen_baseline_batch)
            new_batch = data.union(gen_baseline_output)
            reward_baseline_tensor, _ = self.reward.compute_rewards(new_batch)
            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
            new_batch.batch["reward_baselines"] = reward_baseline_tensor
            del gen_baseline_batch, gen_baseline_output
        
        # repeat to align with repeated responses in rollout
        data = data.repeat(repeat_times=self.config.n, interleave=True)
        new_batch = data.union(gen_batch_output)

        # compute rewards first
        reward_tensor, reward_metrics = self.reward.compute_rewards(new_batch)
        new_batch.batch["token_level_scores"] = reward_tensor
        # store per-sample reward metrics into batch for later slicing/aggregation
        for k, v in reward_metrics.items():
            new_batch.batch[f"reward_{k}"] = torch.tensor(v, dtype=torch.float32, device=reward_tensor.device)

        new_batch = filter(new_batch)
        return new_batch
