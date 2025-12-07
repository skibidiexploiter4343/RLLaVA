import os
import time
import torch
import requests
import numpy as np
from typing import Optional, Iterable, Tuple, Dict, Any, Union, TYPE_CHECKING
from rllava.data.protocol import DataProto
from transformers import PreTrainedTokenizer, ProcessorMixin
from rllava.data.data_utils import process_image, process_video
if TYPE_CHECKING:
    # Import for type checking only to avoid runtime circular import
    from rllava.ppo.config import RolloutConfig



def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    # repeat the elements, supports both tensor and numpy array
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)

def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[Dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None
    
def _process_multi_modal_data(
    multi_modal_data: Dict[str, Any], min_pixels: int, max_pixels: int, video_fps: float, processor: Optional[ProcessorMixin] = None
) -> Dict[str, Any]:
    # may convert image path to image object
    images, videos = [], []
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels, processor))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            videos.append(process_video(video, min_pixels, max_pixels, video_fps))

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None

class InferenceEngine():
    """Base class for all inference engines.
    
    This class defines the common interface that all rollout engines must implement.
    Engines are responsible for managing the rollout process for PPO training.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: 'RolloutConfig',
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        self.model = model
        # Keep tokenizer and processor references for engines needing decoding or multimodal preprocessing
        self.tokenizer = tokenizer
        self.processor = processor
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        if (torch.distributed.is_initialized() and config.tensor_parallel_size > torch.distributed.get_world_size()):
            raise ValueError("Tensor parallelism size should be less than world size.")

        self.engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            self.engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                self.engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        tp_size = self.config.tensor_parallel_size
        dp_size = self.world_size // tp_size if tp_size > 0 else 1
        if self.world_size % tp_size != 0:
            raise ValueError(f"rollout world size {self.world_size} is not divisible by tp size {tp_size}.")

        # Avoid creating distributed device mesh here since engine runs only on rank0.
        # Creating a mesh would require collectives across all ranks and can deadlock.
        self.device_mesh = None

    def check_health(self, base_url):
        # Check server endpoint
        try:
            response = requests.get(f"{base_url}/health", timeout=30)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            return False
        
    def wait_for_server(self, address):
        base_url = f"http://{address}"
        tik = time.time()
        while time.time() - tik < self.config.setup_timeout:
            if self.check_health(base_url):
                return
            time.sleep(1)
        raise RuntimeError("server launch failed")

    def generate(self, prompts: DataProto) -> DataProto:
        raise NotImplementedError()

    async def agenerate(self, prompts: DataProto) -> DataProto:
        """Asynchronously generate a response for the given request."""
        raise NotImplementedError()

    def load(self, model):
        """Load the engine and sync the weights."""
        raise NotImplementedError()

    def offload(self):
        """Offload the engine."""
        raise NotImplementedError()        
    

    def set_version(self, version: int) -> None:
        """Set the current weight version in the inference engine."""
        raise NotImplementedError()

    def get_version(self) -> int:
        """Get the current weight version in the inference engine."""
        raise NotImplementedError()
    
    def pause(self):
        """Pause request submission for async rollout. Used during evaluation to prevent data over generation."""
        raise NotImplementedError()

    def resume(self):
        """Resume request submission for async rollout."""
        raise NotImplementedError()