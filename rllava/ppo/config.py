import os
from typing import Any, Dict, Optional
from dataclasses import asdict, dataclass, field
from rllava.utils.config import BaseConfig, ModelConfig, OptimConfig, FSDPConfig, DeepSpeedConfig, CheckpointConfig
from rllava.engine import VLLMConfig, SGLangConfig
    


@dataclass
class RolloutConfig:
    name: str = "vllm"  # options: vllm, sglang
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    tensor_parallel_size: int = 1
    trust_remote_code: bool = True
    seed: int = 1
    limit_images: int = 0
    ignore_eos: bool = False
    disable_tqdm: bool = False
    micro_batch_size: int = 2
    val_override_config: Dict[str, Any] = field(default_factory=dict)
    # service mode (inference rollout decoupled as HTTP service)
    service_mode: bool = False
    service_url: Optional[str] = None
    service_timeout: float = 60.0
    # below are auto keys
    prompt_length: int = field(default=-1, init=False)
    response_length: int = field(default=-1, init=False)
    adv_estimator: str = field(default="grpo", init=False)
    min_pixels: Optional[int] = field(default=None, init=False)
    max_pixels: Optional[int] = field(default=None, init=False)
    video_fps: float = field(default=2.0, init=False)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    sglang: SGLangConfig = field(default_factory=SGLangConfig)
    load_format: str = "dummy"
    def to_dict(self):
        return asdict(self)


@dataclass
class RewardConfig:
    reward_type: str = "batch"
    reward_function: Optional[str] = None
    reward_function_kwargs: dict = field(default_factory=dict)
    skip_special_tokens: bool = True
    num_cpus: int = 1
    # below are auto keys
    reward_function_name: Optional[str] = field(default=None, init=False)
    model: ModelConfig = field(default_factory=ModelConfig)

    def post_init(self):
        if self.reward_function is not None:  # support custom reward function, e.g., ./math.py:main
            if ":" not in self.reward_function:
                self.reward_function_name = "main"
            else:
                self.reward_function, self.reward_function_name = self.reward_function.rsplit(":", maxsplit=1)

            if os.path.exists(self.reward_function):  # ray job uses absolute path
                self.reward_function = os.path.abspath(self.reward_function)
            else:
                print(f"Reward function {self.reward_function} not found.")
                self.reward_function = None


@dataclass
class CriticConfig:
    strategy: str = "fsdp"
    ppo_mini_batch_size: int = 256
    """number of samples per minibatch for updating critic"""
    ppo_micro_batch_size_per_gpu: int = 4
    """number of samples per forward pass for updating critic"""
    log_prob_micro_batch_size_per_gpu: int = 16
    """number of samples per forward pass for computing values"""
    max_grad_norm: float = 1.0
    """number to clip grad norm"""
    cliprange_value: float = 0.5
    """clip range for value loss"""
    loss_avg_mode: str = "token"
    """loss average mode: `token`, `seq`"""
    ppo_epochs: int = 1
    """number of ppo epochs for each rollout batch"""
    padding_free: bool = False
    """use padding-free training"""
    dynamic_batching: bool = True
    """enable dynamic batching"""
    ulysses_size: int = 1
    """ulysses sequence parallel size"""
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)
    # below are auto keys
    global_batch_size_per_device: int = field(default=-1, init=False)


@dataclass
class PolicyLossConfig:
    """Configuration for policy loss computation.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        loss_mode (str): Loss function mode. Options: 'vanilla', 'clip-cov', 'kl-cov', 'gpg'.
        clip_cov_ratio (float): Ratio of tokens to be clipped for clip-cov loss.
        clip_cov_lb (float): Lower bound for clip-cov loss.
        clip_cov_ub (float): Upper bound for clip-cov loss.
        kl_cov_ratio (float): Ratio of tokens to be applied KL penalty for kl-cov loss.
        ppo_kl_coef (float): KL divergence penalty coefficient.
    """

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1


@dataclass
class ActorConfig:
    strategy: str = "fsdp"
    ppo_mini_batch_size: int = 256
    """number of samples per minibatch for updating actor"""
    ppo_micro_batch_size_per_gpu: int = 4
    """number of samples per forward pass for updating actor"""
    log_prob_micro_batch_size_per_gpu: int = 16
    """number of samples per forward pass for computing log probs"""
    max_grad_norm: float = 1.0
    """number to clip grad norm"""
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    """clip ratio in PPO & DAPO"""
    clip_ratio_high: float = 0.2
    """clip ratio in PPO & DAPO"""
    clip_ratio_c: float = 3.0
    """constant C in dual-clip PPO, clips when advantage < -C"""
    loss_agg_mode: str = "token-mean"
    """loss aggregate mode: `token-mean`, `seq-mean-token-sum`"""
    entropy_coeff: float = 0
    tis_imp_ratio_cap: float = -1
    ppo_epochs: int = 1
    """number of ppo epochs for each rollout batch"""
    padding_free: bool = True
    """use padding-free training"""
    dynamic_batching: bool = True
    """enable dynamic batching"""
    ulysses_size: int = 1
    """ulysses sequence parallel size"""
    use_torch_compile: bool = True
    # UFT-style joint SFT loss (cross-entropy over ground_truth)
    sft_loss_coef: float = 0.0
    model: ModelConfig = field(default_factory=ModelConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)
    global_batch_size_per_device: int = field(default=-1, init=False)
    use_kl_loss: bool = True
    use_kl_in_reward: bool = False
    kl_loss_coef: float = 0.01
    kl_loss_type: str = "low_var_kl"


@dataclass
class RefConfig:
    strategy: str = "fsdp"
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)
    # below are auto keys
    log_prob_micro_batch_size_per_gpu: int = field(default=-1, init=False)
    padding_free: bool = field(default=False, init=False)
    dynamic_batching: bool = field(default=False, init=False)
    ulysses_size: int = field(default=1, init=False)
    use_torch_compile: bool = field(default=True, init=False)


@dataclass
class KLControlConfig:
    """Configuration for KL control.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        type (str): Type of KL control. Can be "fixed" or "adaptive".
        kl_coef (float): Initial coefficient for KL penalty.
        horizon (int): Horizon value for adaptive controller.
        target_kl (float): Target KL divergence for adaptive controller.
    """

    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: int = 10000
    target_kl: float = 0.1


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator: `gae`, `grpo`, `reinforce_plus_plus`, `remax`, `rloo`, etc"""
    norm_adv_by_std_in_grpo: bool = True
    """whether to normalize advantage by standard deviation in grpo"""
    kl_penalty: str = "kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)
    """kl controller for reward calculation"""
    use_kl_in_reward: bool = False
    """whether to use kl penalty in reward calculation"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""    


@dataclass
class PPOConfig(BaseConfig):
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: Optional[CriticConfig] = None
    ref: RefConfig = field(default_factory=RefConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)

    def post_init(self):
        self.rollout.prompt_length = self.data.max_prompt_length
        self.rollout.response_length = self.data.max_response_length
        self.rollout.trust_remote_code = self.actor.model.trust_remote_code
        self.rollout.adv_estimator = self.algorithm.adv_estimator
        self.rollout.min_pixels = self.data.min_pixels
        self.rollout.max_pixels = self.data.max_pixels
        self.rollout.video_fps = self.data.video_fps
        self.actor.use_kl_in_reward = self.algorithm.use_kl_in_reward
        self.ref.log_prob_micro_batch_size_per_gpu = self.actor.log_prob_micro_batch_size_per_gpu
        self.ref.padding_free = self.actor.padding_free
        self.ref.ulysses_size = self.actor.ulysses_size
        self.ref.use_torch_compile = self.actor.use_torch_compile
        self.model_path = self.actor.model.model_path

