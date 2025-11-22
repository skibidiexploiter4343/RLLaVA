import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Dict, List



@dataclass
class ModelConfig:
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    torch_dtype: Optional[str] = None
    tie_word_embeddings: bool = field(default=False)
    attn_implementation: Optional[str] = field(default='flash_attention_2')
    override_config: Dict[str, Any] = field(default_factory=dict)
    enable_gradient_checkpointing: bool = True
    enable_activation_offload: bool = False
    trust_remote_code: bool = True
    freeze_vision_tower: bool = False
    padding_free: bool = True
    dynamic_batching: bool = True
    ulysses_size: int = 1
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
    tune_embed_tokens: bool = field(default=False)
    
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