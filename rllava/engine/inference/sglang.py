from __future__ import annotations

import logging
import os
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
import torch
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin

from .base import InferenceEngine
from .. import register_engine
from rllava.data.protocol import DataProto
from rllava.utils import torch_functional as VF
# from .remote_utils import ServiceProcess, ensure_port, find_free_port
from .config import SGLangConfig
try:
    import sglang as sgl
except Exception:
    sgl = None


@register_engine("sglang")
class SGLangEngine(InferenceEngine):
    """High-level rollout engine powered by SGLang."""

    def __init__(
        self,
        model_name_or_path: str,
        config: "RolloutConfig",
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        engine_args: Optional[Dict] = None,
    ) -> None:
        super().__init__(model_name_or_path, config, tokenizer, processor)
        self.engine_args = engine_args or {}

        self._engine = None
        self._logger = logging.getLogger(__file__)
        self._version = 0
        self._workflow_executor = None
        self._engine_id: Optional[str] = None
        self._loaded = False
        self._service_process: Optional[ServiceProcess] = None
        self._service_base_url: Optional[str] = None
        self._service_session: Optional[requests.Session] = None
        self._service_generate_endpoint: Optional[str] = None
        self._service_timeout = config.service_timeout
        self._service_headers: Dict[str, str] = {}
        self.service_mode = config.service_mode

        if self.service_mode:
            self._initialize_remote_service(model_name_or_path)
        else:
            self._ensure_backend_available()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_backend_available(self) -> None:
        if sgl is None:
            raise RuntimeError(
                "SGLang is not installed or failed to import. "
                "Please ensure `sglang>=0.4.9.post2` is available before using the SGLang inference engine."
            )

    def _build_engine_kwargs(self) -> Dict:
        """Merge config-driven runtime arguments with user overrides."""
        cfg_dict = asdict(self.config.sglang) if hasattr(self.config, "sglang") else {}
        if cfg_dict.get("model_path") in (None, ""):
            cfg_dict["model_path"] = self.model
        cfg_dict = {k: v for k, v in cfg_dict.items() if v is not None}
        merged = {**cfg_dict, **self.engine_args}
        return merged

    def _create_engine(self) -> "sgl.Engine":  # type: ignore[name-defined]
        engine_kwargs = self._build_engine_kwargs()
        self._logger.info("Initializing SGLang engine with args: %s", {k: engine_kwargs[k] for k in sorted(engine_kwargs)})
        try:
            return sgl.Engine(**engine_kwargs)
        except Exception as exc:  # pragma: no cover - pass-through to caller with context
            self._logger.exception("Failed to initialize SGLang engine", exc_info=exc)
            raise

    def _initialize_remote_service(self, model_name_or_path: str) -> None:
        service_cfg = deepcopy(self.config.service)

        host = service_cfg.host or "127.0.0.1"
        address: Optional[str] = None

        if self.config.service_url:
            base_url = self._normalize_service_url(self.config.service_url)
            self._service_base_url = base_url.rstrip("/")
            parsed = urlparse(self._service_base_url)
            address = parsed.netloc or parsed.path
            if address:
                self.wait_for_server(address)
        else:
            port = ensure_port(service_cfg.port, host=host)
            address = f"{host}:{port}"

        should_launch_service = (self.rank == 0)

        if self.config.service_url or service_cfg.reuse_existing:
            if address:
                self.wait_for_server(address)
            if self._service_base_url is None and address:
                self._service_base_url = f"http://{address}"
        else:
            if should_launch_service:
                dist_addr = service_cfg.dist_init_addr or f"{host}:{find_free_port(host=host)}"
                sglang_cfg = deepcopy(self.config.sglang)
                if not sglang_cfg.model_path:
                    sglang_cfg.model_path = model_name_or_path

                cmd = SGLangConfig.build_cmd(
                    sglang_config=sglang_cfg,
                    tp_size=self.config.tensor_parallel_size,
                    base_gpu_id=0,
                    host=host,
                    port=port,
                    dist_init_addr=dist_addr,
                )
                cmd = cmd.replace("\\\n", " ").replace("\\", " ")

                env = deepcopy(service_cfg.env) if service_cfg.env else {}
                if "CUDA_VISIBLE_DEVICES" not in env:
                    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                log_path = service_cfg.log_path or os.path.join(os.getcwd(), "logs", "sglang_service.log")

                self._service_process = ServiceProcess(
                    command=cmd,
                    env=env,
                    log_path=log_path,
                ).start()

            if address:
                self.wait_for_server(address)
            self._service_base_url = f"http://{address}"

        if self._service_base_url is None:
            raise RuntimeError("Failed to determine remote service base URL for SGLangEngine.")

        self._service_session = requests.Session()
        self._service_headers = {"Content-Type": "application/json"}
        self._service_session.headers.update(self._service_headers)
        self._service_generate_endpoint = urljoin(self._service_base_url + "/", "generate")
        self.config.service_url = self._service_base_url

    def _normalize_service_url(self, url: str) -> str:
        if not url:
            raise ValueError("Empty service URL provided.")
        normalized = url.strip()
        if not normalized.startswith("http://") and not normalized.startswith("https://"):
            normalized = f"http://{normalized}"
        parsed = urlparse(normalized)
        if not parsed.netloc:
            raise ValueError(f"Invalid service URL: {url}")
        return normalized

    def _prepare_sampling_params(self, meta_info: Dict[str, float | int | bool]) -> Dict:
        """Build request-level sampling params using rollout defaults plus overrides from meta_info."""
        params = {
            "n": 1,
            "temperature": meta_info.get("temperature", self.config.temperature),
            "top_p": meta_info.get("top_p", getattr(self.config, "top_p", 1.0)),
            "top_k": meta_info.get("top_k", getattr(self.config, "top_k", -1)),
            "max_new_tokens": meta_info.get("max_new_tokens", self.config.response_length),
            "presence_penalty": meta_info.get("presence_penalty", 0.0),
            "frequency_penalty": meta_info.get("frequency_penalty", 0.0),
            "ignore_eos": meta_info.get("ignore_eos", self.config.ignore_eos),
        }
        if params["temperature"] == 0:
            params.update({
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
            })
        return params

    def _ensure_raw_prompt_ids(self, prompts: DataProto) -> np.ndarray:
        """Ensure `raw_prompt_ids` exist in non-tensor batch, deriving from input_ids if necessary."""
        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" in non_tensor_batch:
            return non_tensor_batch["raw_prompt_ids"]

        input_ids: torch.Tensor = prompts.batch["input_ids"].to("cpu")
        attention_mask: torch.Tensor = prompts.batch["attention_mask"].to("cpu")
        pad_token_id = prompts.meta_info.get("pad_token_id", self.pad_token_id)
        raw_prompt_ids = []
        for seq, mask in zip(input_ids, attention_mask, strict=True):
            valid_length = int(mask.sum().item())
            tokens = seq[-valid_length:].tolist()
            if pad_token_id is not None:
                tokens = [tid for tid in tokens if tid != pad_token_id]
            raw_prompt_ids.append(tokens)
        raw_prompt_ids = np.array(raw_prompt_ids, dtype=object)
        non_tensor_batch["raw_prompt_ids"] = raw_prompt_ids
        return raw_prompt_ids

    def _prepare_multi_modal_entries(self, prompts: DataProto, batch_size: int) -> Optional[np.ndarray]:
        non_tensor_batch = prompts.non_tensor_batch
        if "multi_modal_data" in non_tensor_batch:
            return non_tensor_batch["multi_modal_data"]
        if self.processor is None:
            return None
        if "multi_modal_inputs" not in non_tensor_batch:
            return None

        mm_array = np.array(non_tensor_batch["multi_modal_inputs"], dtype=object)
        non_tensor_batch["multi_modal_data"] = mm_array
        return mm_array

    def _build_request_payloads(self, prompts: DataProto) -> List[Dict]:
        """Convert `DataProto` prompts into SGLang request payloads."""
        batch_size = len(prompts)
        raw_prompt_ids = self._ensure_raw_prompt_ids(prompts)
        multi_modal_entries = self._prepare_multi_modal_entries(prompts, batch_size)
        agent_names = prompts.non_tensor_batch.get("agent_name")
        tools_kwargs = prompts.non_tensor_batch.get("tools_kwargs")
        payloads: List[Dict] = []
        for idx in range(batch_size):
            payload: Dict = {"prompt_token_ids": list(map(int, raw_prompt_ids[idx]))}
            if multi_modal_entries is not None:
                payload["multi_modal_data"] = multi_modal_entries[idx]
            if agent_names is not None:
                payload["agent_name"] = agent_names[idx]
            if tools_kwargs is not None:
                payload["tools_kwargs"] = tools_kwargs[idx]
            payloads.append(payload)
        return payloads

    def _post_remote(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._service_session is None or self._service_generate_endpoint is None:
            raise RuntimeError("Remote SGLang service endpoint not initialized.")
        response = self._service_session.post(
            self._service_generate_endpoint,
            json=payload,
            timeout=self._service_timeout,
        )
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    def initialize(self, engine_id: Optional[str] = None) -> None:
        if self._engine is not None:
            self._logger.debug("SGLang engine already initialized; skipping re-init.")
            return

        self._ensure_backend_available()
        self._engine_id = engine_id or "sglang"
        self._logger = logging.getLogger(f"rllava.sglang.engine[{self._engine_id}]")
        self._engine = self._create_engine()
        self._logger.info("SGLang engine initialized.")

    def destroy(self) -> None:
        if self._engine is None:
            return
        try:
            shutdown = getattr(self._engine, "shutdown", None)
            if callable(shutdown):
                shutdown()
        finally:
            self._engine = None
            self._logger.info("SGLang engine destroyed.")

    def load(self, model) -> None:
        if self.service_mode:
            if self._service_base_url is None:
                raise RuntimeError("Remote service base URL not initialized for SGLangEngine.")
            parsed = urlparse(self._service_base_url)
            address = parsed.netloc or parsed.path
            if address:
                self.wait_for_server(address)
            self._loaded = True
            return

        if not self.is_initialized:
            self.initialize()

        if self._loaded:
            self._logger.debug("SGLang engine already loaded; skipping.")
            return

        self._logger.warning(
            "SGLangEngine.load currently acts as a no-op. Ensure remote weights"
            " are refreshed through external mechanisms if required."
        )
        self._loaded = True

    def offload(self) -> None:
        if self.service_mode:
            if self._service_session is not None:
                self._service_session.close()
                self._service_session = None
            if self._service_process is not None:
                self._service_process.terminate()
                self._service_process = None
            self._loaded = False
            return

        if not self._loaded:
            return
        self._logger.debug("SGLang engine offload (no-op).")
        self._loaded = False

    def update_weights(self, model) -> None:
        # Placeholder until SGLang weight API is integrated.
        self._logger.warning(
            "SGLangEngine.update_weights is currently a stub."
        )

    def set_version(self, version: int) -> None:
        self._version = version

    def get_version(self) -> int:
        return self._version

    @property
    def is_initialized(self) -> bool:
        return self._engine is not None

    # ------------------------------------------------------------------
    # Generation interfaces
    # ------------------------------------------------------------------
    def generate(self, prompts: DataProto) -> DataProto:
        if self.service_mode:
            return self._generate_remote(prompts)

        if not self.is_initialized:
            self.initialize()

        input_ids: torch.Tensor = prompts.batch["input_ids"]
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info.get("eos_token_id", self.tokenizer.eos_token_id)

        sampling_params_dict = self._prepare_sampling_params(prompts.meta_info)

        payloads = self._build_request_payloads(prompts)

        responses = [
            self._engine.generate(
                sampling_params=sampling_params_dict,
                input_ids=payload.get("prompt_token_ids"),
                image_data=payload.get("multi_modal_data"),
                return_logprob=True,
            )
            for payload in payloads
        ]

        response_ids_list: List[List[int]] = []
        response_logprobs_list: List[List[float]] = []
        stop_reasons: List[str] = []

        for result in responses:
            meta_info = result.get("meta_info", {})
            token_entries = meta_info.get("output_token_logprobs", [])

            tokens: List[int] = []
            logprobs: List[float] = []
            for entry in token_entries:
                if isinstance(entry, dict):
                    token_value = entry.get("token_id") or entry.get("token") or entry.get("value")
                    logprob_value = entry.get("logprob") or entry.get("score") or entry.get("log_prob")
                else:
                    token_value = entry[1] if len(entry) > 1 else None
                    logprob_value = entry[0]
                if token_value is None:
                    continue
                tokens.append(int(token_value))
                logprobs.append(float(logprob_value) if logprob_value is not None else 0.0)

            response_ids_list.append(tokens)
            response_logprobs_list.append(logprobs)

            finish_reason = meta_info.get("finish_reason", {})
            stop_reasons.append(finish_reason.get("type", finish_reason or "length"))

        max_response_len = self.config.response_length
        pad_token_id = self.pad_token_id if self.pad_token_id is not None else self.tokenizer.pad_token_id

        def _pad_sequence(seq: List[int], pad_value: int) -> List[int]:
            truncated = seq[:max_response_len]
            padding = [pad_value] * (max_response_len - len(truncated))
            return truncated + padding

        def _pad_logprobs(seq: List[float]) -> List[float]:
            truncated = seq[:max_response_len]
            padding = [0.0] * (max_response_len - len(truncated))
            return truncated + padding

        padded_responses = [_pad_sequence(tokens, pad_token_id) for tokens in response_ids_list]
        padded_logprobs = [_pad_logprobs(logprobs) for logprobs in response_logprobs_list]

        responses_tensor = torch.tensor(padded_responses, dtype=input_ids.dtype, device=input_ids.device)
        logprobs_tensor = torch.tensor(padded_logprobs, dtype=torch.float32, device=input_ids.device)

        sequence_ids = torch.cat([input_ids, responses_tensor], dim=-1)
        response_len = responses_tensor.size(1)

        delta_position_id = torch.arange(1, response_len + 1, device=position_ids.device)
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(1, 1, -1).expand(position_ids.size(0), position_ids.size(1), -1)
        else:
            delta_position_id = delta_position_id.view(1, -1).expand(position_ids.size(0), -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_mask = VF.get_response_mask(response_ids=responses_tensor, eos_token_id=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        batch = {
            "prompts": input_ids,
            "responses": responses_tensor,
            "input_ids": sequence_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": position_ids,
            "response_logprobs": logprobs_tensor,
        }

        stop_reasons_np = np.array(stop_reasons, dtype=object)
        non_tensor_batch = {"stop_reasons": stop_reasons_np}

        if "multi_modal_data" in prompts.non_tensor_batch:
            non_tensor_batch["multi_modal_data"] = prompts.non_tensor_batch["multi_modal_data"]

        meta_info = dict(prompts.meta_info)
        meta_info["stop_reasons"] = stop_reasons

        batch_collated = TensorDict(batch, batch_size=responses_tensor.size(0))

        return DataProto(
            batch=batch_collated,
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )

    def _generate_remote(self, prompts: DataProto) -> DataProto:
        if self._service_session is None or self._service_base_url is None:
            raise RuntimeError("Remote SGLang service session not initialized.")

        input_ids: torch.Tensor = prompts.batch["input_ids"]
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info.get("eos_token_id", self.tokenizer.eos_token_id)
        batch_size = input_ids.size(0)

        payloads = self._build_request_payloads(prompts)
        sampling_params_dict = self._prepare_sampling_params(prompts.meta_info)

        responses: List[Dict[str, Any]] = []
        for payload in payloads:
            data = {
                "sampling_params": sampling_params_dict,
                "input_ids": payload.get("prompt_token_ids"),
                "image_data": payload.get("multi_modal_data"),
                "return_logprob": True,
            }
            result = self._post_remote(data)
            responses.append(result)

        response_ids_list: List[List[int]] = []
        response_logprobs_list: List[List[float]] = []
        stop_reasons: List[str] = []

        for result in responses:
            meta_info = result.get("meta_info", {})
            token_entries = meta_info.get("output_token_logprobs", [])

            tokens: List[int] = []
            logprobs: List[float] = []
            for entry in token_entries:
                if isinstance(entry, dict):
                    token_value = entry.get("token_id") or entry.get("token") or entry.get("value")
                    logprob_value = entry.get("logprob") or entry.get("score") or entry.get("log_prob")
                else:
                    token_value = entry[1] if len(entry) > 1 else None
                    logprob_value = entry[0]
                if token_value is None:
                    continue
                tokens.append(int(token_value))
                logprobs.append(float(logprob_value) if logprob_value is not None else 0.0)

            response_ids_list.append(tokens)
            response_logprobs_list.append(logprobs)

            finish_reason = meta_info.get("finish_reason", {})
            stop_reasons.append(finish_reason.get("type", finish_reason or "length"))

        max_response_len = self.config.response_length
        pad_token_id = self.pad_token_id if self.pad_token_id is not None else self.tokenizer.pad_token_id

        padded_responses = [
            (tokens[:max_response_len] + [pad_token_id] * (max_response_len - len(tokens)))
            for tokens in response_ids_list
        ]
        padded_logprobs = [
            (logprobs[:max_response_len] + [0.0] * (max_response_len - len(logprobs)))
            for logprobs in response_logprobs_list
        ]

        responses_tensor = torch.tensor(padded_responses, dtype=input_ids.dtype, device=input_ids.device)
        logprobs_tensor = torch.tensor(padded_logprobs, dtype=torch.float32, device=input_ids.device)

        sequence_ids = torch.cat([input_ids, responses_tensor], dim=-1)
        response_len = responses_tensor.size(1)

        delta_position_id = torch.arange(1, response_len + 1, device=position_ids.device)
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(1, 1, -1).expand(position_ids.size(0), position_ids.size(1), -1)
        else:
            delta_position_id = delta_position_id.view(1, -1).expand(position_ids.size(0), -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_mask = VF.get_response_mask(response_ids=responses_tensor, eos_token_id=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        batch = {
            "prompts": input_ids,
            "responses": responses_tensor,
            "input_ids": sequence_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": position_ids,
            "response_logprobs": logprobs_tensor,
        }

        stop_reasons_np = np.array(stop_reasons, dtype=object)
        non_tensor_batch = {"stop_reasons": stop_reasons_np}
        multi_modal_data = prompts.non_tensor_batch.get("multi_modal_data")
        if multi_modal_data is not None:
            non_tensor_batch["multi_modal_data"] = multi_modal_data

        meta_info = dict(prompts.meta_info)
        meta_info["stop_reasons"] = stop_reasons

        batch_collated = TensorDict(batch, batch_size=responses_tensor.size(0))

        return DataProto(
            batch=batch_collated,
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )

    async def agenerate(self, prompts: DataProto):
        raise NotImplementedError("SGLangEngine.agenerate is not implemented yet")

    # ------------------------------------------------------------------
    # Workflow helpers (agentic / multi-turn)
    # ------------------------------------------------------------------
    def submit(self, *args, **kwargs):
        raise NotImplementedError("SGLangEngine.submit is not implemented yet")

    def wait(self, *args, **kwargs):
        raise NotImplementedError("SGLangEngine.wait is not implemented yet")

    def pause(self):
        raise NotImplementedError("SGLangEngine.pause is not implemented yet")

    def resume(self):
        raise NotImplementedError("SGLangEngine.resume is not implemented yet")