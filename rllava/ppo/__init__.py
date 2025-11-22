from .ppo import PPO
from .config import PPOConfig, RolloutConfig
from .role.rollout import Rollout
from .role.reward import Reward
from rllava.ppo.plugins import get_adv_estimator, get_policy_loss



__all__ = [
    "PPOConfig",
    "PPOFactory",
    "PPO",
    "RolloutConfig",
]


class PPOFactory:
    
    @staticmethod
    def build(config: PPOConfig, tokenizer, processor) -> PPO:
        # Assemble plugins/components from config

        adv_estimator = get_adv_estimator(getattr(config.algorithm, "adv_estimator", "grpo")) 

        policy_loss = get_policy_loss(getattr(config.actor.policy_loss, "loss_mode", 'vanilla'))

        reward = Reward(config.reward, tokenizer)
        
        rollout = Rollout(config.rollout, reward, tokenizer, processor)

        # Instantiate PPO and inject components
        return PPO(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            reward=reward,
            rollout=rollout,
            adv_estimator=adv_estimator,
            policy_loss=policy_loss,
        )
