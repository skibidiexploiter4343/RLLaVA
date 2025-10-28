import torch
import os
import numpy as np
import torch.distributed as dist
from contextlib import contextmanager
from collections import defaultdict
from copy import deepcopy
from .config import PPOConfig
from .role.actor import Actor
from .role.critic import Critic
from .role.reward import Reward
from rllava.data.protocol import DataProto
from rllava.utils import torch_functional as VF
from rllava.utils.dist_utils import (
    dist_gather_then_scatter,
    dist_batch,
    gather_batch,
)
from .utils.core_algos import kl_penalty, get_kl_controller
from rllava.utils.logger.aggregate_logger import print_rank_0
from .plugins.rollout import Rollout
from transformers import AutoConfig
from rllava.engine import EngineFactory



class PPO():
    
    def __init__(self, config: PPOConfig, tokenizer, processor, reward: Reward, rollout: Rollout, adv_estimator=None, policy_loss=None):
        self.config = config
        self.train_batch_size = config.data.train_batch_size
        self.tokenizer = tokenizer
        self.processor = processor

        self.model_config = AutoConfig.from_pretrained(
            config.model_path,
            trust_remote_code=config.actor.model.trust_remote_code,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            attn_implementation=config.actor.model.attn_implementation,
            **config.actor.model.override_config,
        )
        print_rank_0(f"Model config: {self.model_config}")

        if config.algorithm.use_kl_in_reward and config.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_critic = self.config.critic is not None
        
        if self.config.data.train_batch_size % self.config.actor.ppo_mini_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")
        
        if (
            self.config.data.train_batch_size * self.config.rollout.n
        ) % self.config.actor.log_prob_micro_batch_size_per_gpu != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )
        
        if config.rollout.n > 1:
            self.config.actor.ppo_mini_batch_size *= config.rollout.n
            print_rank_0(f"Actor will use global batch size {self.config.actor.ppo_mini_batch_size}.")
        
        if self.use_critic:
            if self.config.data.train_batch_size % self.config.critic.ppo_mini_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                self.config.data.train_batch_size * self.config.rollout.n
            ) % self.config.critic.log_prob_micro_batch_size_per_gpu != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        # Injected plugins/components (with sensible fallbacks)
        self.adv_estimator = adv_estimator

        self.actor = Actor(config.actor, 
                           policy_loss=policy_loss,
                           tokenizer=tokenizer,
                           processor=processor)
        self.critic = Critic(config.critic) if config.critic is not None else None

        self.reward = reward
        self.rollout = rollout

        def online_sampling(batch: DataProto) -> DataProto:
            if self.config.algorithm.online_filtering:
                key = f"reward_{self.config.algorithm.filter_key}"
                # convert to python list for numpy ops
                filter_scores = batch.batch[key].detach().cpu().tolist()
                uids = batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)
                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                batch = batch[kept_sample_idxs]
            return batch
        
        self.filter = online_sampling

    def initialize(self, train_dataloader):
        self.actor.initialize(self.model_config)
        self.reward.initialize()

        self.rollout.initialize(self.config.actor.model.model_path)
        
        self.training_steps = self.get_training_steps(train_dataloader)
        self.config.actor.optim.training_steps = self.training_steps
        if self.use_critic:
            self.config.critic.optim.training_steps = self.training_steps
            self.critic.initialize()

    def get_training_steps(self, train_dataloader):  
        if self.config.trainer.max_steps is not None:
            return self.config.trainer.max_steps
        else:
            return len(train_dataloader) * self.config.trainer.total_epochs

    def load_checkpoint(self, checkpoint_path: str):
        actor_path = os.path.join(checkpoint_path, "actor")
        self.actor.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(checkpoint_path, "critic")
            self.critic.load_checkpoint(critic_path)

    def save_checkpoint(self, checkpoint_path: str):
        actor_path = os.path.join(checkpoint_path, "actor")
        self.actor.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)
        if self.use_critic:
            critic_path = os.path.join(checkpoint_path, "critic")
            self.critic.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

    def rollout_batch(self, data_iterator) -> DataProto:
        batch = None
        num_try_make_batch = 0
        print_rank_0("Start generating batch...")
        with self.generate_context():
            while True:
                num_try_make_batch += 1
                batch_in = dist_batch(data_iterator)

                new_batch = self.rollout.generate_one_batch(batch_in, self.filter)
                # print('before gather', len(new_batch))
                new_batch = gather_batch(new_batch)
                # print('after gather', len(new_batch))
                batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
                current_batch_size = len(batch) // self.rollout.n
                rollout_batch_size = self.train_batch_size
                if current_batch_size < rollout_batch_size:
                    print_rank_0(f"{current_batch_size=} < {rollout_batch_size=}")
                    max_try_make_batch = self.config.trainer.max_try_make_batch
                    if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                        print_rank_0(f"{num_try_make_batch=}. Continue generating...")
                    else:
                        raise ValueError(f"{num_try_make_batch=} >= {max_try_make_batch=}." + " Generated too many. Please check if your data are too difficult." + " You could also try set max_num_gen_batches=0 to enable endless trials.")
                else:
                    print_rank_0(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                    result = batch[: self.train_batch_size * self.rollout.n]
                    result.meta_info["global_token_num"] = torch.sum(result.batch["attention_mask"], dim=-1).tolist()
                    result = result.chunk(dist.get_world_size())[dist.get_rank()]
                    return result
        
    @contextmanager
    def generate_context(self):
        
        with self.actor.unwrap_model_for_generation() as unwrapped_model:
            self.rollout.rollout_engine.load(unwrapped_model)

        yield
        self.rollout.rollout_engine.offload()

    def compute_values(self, batch):
        return self.critic.compute_values(batch)

    def compute_rewards(self, batch):
        return self.reward.compute_rewards(batch)

    def compute_log_probs(self, data: DataProto):
        batch_size = len(data)
        num_mini_batches = (batch_size + self.config.actor.global_batch_size_per_device - 1) // self.config.actor.global_batch_size_per_device
        on_policy = num_mini_batches == 1 and self.config.actor.ppo_epochs == 1

        if not on_policy:
            old_log_probs = self.compute_old_log_probs(data)
            data = data.union(old_log_probs)

        # compute ref_log_probs if needed
        if self.config.actor.use_kl_loss or self.config.algorithm.use_kl_in_reward:
            ref_log_probs = self.compute_ref_log_probs(data)
            data = data.union(ref_log_probs)
        return data

    def compute_old_log_probs(self, data: DataProto):
        output = self.actor.compute_log_probs(data, temperature=self.config.rollout.temperature)
        return DataProto.from_dict(
                tensors={"old_log_probs": output}, 
                meta_info={"temperature": self.config.rollout.temperature}
            )

    def compute_ref_log_probs(self, data: DataProto):
        output = self.actor.compute_log_probs(data, temperature=self.config.rollout.temperature, is_ref=True)
        # Wrap result in DataProto with appropriate tensor key
        return DataProto.from_dict(tensors={"ref_log_probs": output})
            
    def apply_kl_penalty(self, data: DataProto, kl_ctrl, kl_p="kl"):
        """Apply KL penalty to the token-level rewards.

        This function computes the KL divergence between the reference policy and current policy,
        then applies a penalty to the token-level rewards based on this divergence.
    
        Args:
            data (DataProto): The data containing batched model outputs and inputs.
            kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
            kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
    
        Returns:
            tuple: A tuple containing:
                - The updated data with token-level rewards adjusted by KL penalty
                - A dictionary of metrics related to the KL penalty
        """
        response_mask = data.batch["response_mask"]
        token_level_scores = data.batch["token_level_scores"]
        batch_size = data.batch.batch_size[0]
        

        kld = kl_penalty(
            data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_p)
        kld = kld * response_mask
        beta = kl_ctrl.value

        data.batch["token_level_rewards"] = token_level_scores - beta * kld

        current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)
        current_kl = torch.mean(current_kl, dim=0).item()
        
        kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
        metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}
        
        return data, metrics
    
    @dist_gather_then_scatter()
    def compute_advantage(self, data: DataProto):
        adv_metrics = {}
        # apply kl penalty if needed
        if self.config.algorithm.use_kl_in_reward:
            data, kl_metrics = self.apply_kl_penalty(
                data, self.kl_ctrl_in_reward, self.config.algorithm.kl_penalty)
            adv_metrics.update(kl_metrics)
        else:
            data.batch["token_level_rewards"] = data.batch["token_level_scores"]

        advantages, returns = self.adv_estimator(data, self.config.algorithm)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data, adv_metrics
    
    def update_model(self, data: DataProto, global_steps) -> DataProto:
        outputs = []
        if self.use_critic:
            critic_output = self.update_critic(data)
            outputs.append(critic_output)

        if self.config.trainer.critic_warmup <= global_steps:
            actor_output = self.update_actor(data)
            outputs.append(actor_output)

        if len(outputs) == 0:
            return DataProto()
        merged = outputs[0]
        for out in outputs[1:]:
            merged = merged.union(out)
        return merged
    
    def update_critic(self, data: DataProto):
        return self.critic.update(data)
    
    def update_actor(self, data: DataProto):
        output = self.actor.update(data)
        return output
