import os
import warnings
from rllava.data.config import DataConfig
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List
from omegaconf import OmegaConf, DictConfig
from typing import Type, TypeVar, cast
from dataclasses import is_dataclass, fields, asdict



T = TypeVar("T")

def init_config(cls: Type[T]) -> T:
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(cls())
    if "config" in cli_args:
        config_path = cli_args["config"]
        del cli_args["config"]
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)
    config = OmegaConf.merge(default_config, cli_args)
    obj = OmegaConf.to_object(config)
    if not isinstance(obj, cls):
        obj = cls(**obj)  # type: ignore[arg-type]
    if hasattr(obj, "deep_post_init"):
        obj.deep_post_init()
    return cast(T, obj)

def conf_as_dict(cfg):
    if isinstance(cfg, (OmegaConf, DictConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    return asdict(cfg)

def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class TrainerConfig:
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "rllava"
    """project name for logger"""
    experiment_name: str = "agent"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "tensorboard")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = True
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation"""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    outputs_dir: Optional[str] = None
    """outputs directory"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""
    find_last_checkpoint: bool = True
    """automatically find the last checkpoint in the save checkpoint path to resume training"""
    async_training: bool = False
    """enable async-like rollout using workflow interface"""

    def post_init(self):
        if self.outputs_dir is None:
            self.outputs_dir = os.path.join(self.project_name, self.experiment_name)
        else:
            self.outputs_dir = os.path.join(self.outputs_dir, self.project_name, self.experiment_name)
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join(self.outputs_dir, "checkpoints")

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # ray job uses absolute path
        if self.load_checkpoint_path is not None:
            if os.path.exists(self.load_checkpoint_path):  # ray job uses absolute path
                self.load_checkpoint_path = os.path.abspath(self.load_checkpoint_path)
            else:
                print(f"Model checkpoint {self.load_checkpoint_path} not found.")
                self.load_checkpoint_path = None


@dataclass
class BaseConfig:
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)

                
@dataclass
class ModelConfig:
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    torch_dtype: Optional[str] = None
    attn_implementation: Optional[str] = field(default='flash_attention_2')
    override_config: Dict[str, Any] = field(default_factory=dict)
    enable_gradient_checkpointing: bool = True
    enable_activation_offload: bool = False
    trust_remote_code: bool = True
    freeze_vision_tower: bool = False
    use_peft: bool = field(
        default=False,
        metadata={"help": "Whether to use PEFT for training."},
    )
    lora_task_type: str = field(default="CAUSAL_LM")
    lora_r: int = field(default=128)
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    lora_alpha: int = field(default=256)
    lora_dropout: float = field(default=0.05)
    lora_use_rslora: bool = field(default=False)
    lora_weight_path: str = ""
    lora_bias: str = "none"
    use_rslora: bool = field(default=False)
    tie_word_embeddings: bool = field(default=False)
    lora_modules_to_save: Optional[List[str]] = field(default=None)

    def post_init(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path

        if self.model_path is not None and os.path.exists(self.model_path):  # ray job uses absolute path
            self.model_path = os.path.abspath(self.model_path)

        if self.tokenizer_path is not None and os.path.exists(self.tokenizer_path):
            self.tokenizer_path = os.path.abspath(self.tokenizer_path)


@dataclass
class LLaVAModelConfig(ModelConfig):
    pretrained_model_path: Optional[str] = None
    cache_dir: Optional[str] = field(default=None)
    vision_tower: Optional[str] = field(default='')
    vision_tower2: Optional[str] = field(default='')
    connector_type: str = field(default='linear')
    
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    resampler_hidden_size: Optional[int] = field(default=768)
    num_queries: Optional[int] = field(default=128)
    num_resampler_layers: Optional[int] = field(default=3)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    tokenizer_use_fast: bool = field(default=False)
    tokenizer_padding_side: str = field(default='right')
    training_recipe: str = field(default='common')
    tune_type_llm: str = field(default="frozen") # support only: frozen, full, lora, qlora_int4, qlora_int8
    tune_type_vision_tower: str = field(default="frozen") # support only: frozen, full, partially-tune
    tune_vision_tower_from_layer: Optional[int] = field(default=10)
    tune_type_connector: str = field(default="full") # support only: frozen, full
    tune_embed_tokens: Optional[int] = field(default=False)
    
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    vision_tower_lr: Optional[float] = None


@dataclass
class CheckpointConfig:
    """Configuration for model checkpointing.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        save_contents (list[str]): What to include in saved checkpoints.
            Options: 'model', 'optimizer', 'extra', 'hf_model'.
        load_contents (list[str]): Contents to load from checkpoint. Defaults to same as save_contents.
        async_save (bool): Whether to save checkpoints asynchronously. Only implemented for Megatron as of now.
    """

    save_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    load_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    async_save: bool = False


@dataclass
class OptimConfig:
    lr: float = 1e-6
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    strategy: str = "adamw"
    lr_warmup_ratio: float = 0.0
    lr_warmup_steps: Optional[int] = None
    lr_scheduler_type: str = "constant"
    min_lr_ratio: Optional[float] = None
    num_cycles: float = 0.5
    warmup_style: str = "constant"
    # below are auto keys
    training_steps: int = field(default=-1, init=False)

    def post_init(self):
        if self.warmup_style is not None:
            assert self.warmup_style in ["constant", "cosine"]
            warnings.warn(
                "`warmup_style` is deprecated, use `lr_scheduler_type` instead.", DeprecationWarning, stacklevel=2
            )
            self.lr_scheduler_type = self.warmup_style
        assert self.lr_scheduler_type in ["constant", "cosine"]


@dataclass
class FSDPConfig:
    wrap_policy: dict[str, Any] = field(default_factory=dict)
    enable_cpu_offload: bool = False
    use_orig_params: bool = False
    torch_dtype: Optional[str] = None
    offload_params: bool = False
    offload_optimizer: bool = False
    forward_prefetch: bool = False
    fsdp_size: int = -1
    ulysses_size: int = 1
    mixed_precision: Optional[dict[str, Any]] = None


@dataclass
class DeepSpeedConfig:
    zero_stage: int = 3
    enable_full_shard: bool = True
    enable_cpu_offload: bool = False
    use_orig_params: bool = False
    torch_dtype: Optional[str] = None
