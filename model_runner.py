from __future__ import annotations

import json
import logging
import os
import re
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

from config import Config
from prompts import get_prompts


LOGGER = logging.getLogger(__name__)
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those",
    "it", "its", "them", "their",
    "video", "scene", "shows", "showing", "describe", "describing", "resulting", "edit",
    "instruction", "instructions", "reference", "subject", "subjects", "action", "actions",
    "camera", "lighting", "background", "atmosphere", "mood", "overall", "then", "now",
    "summary", "primary", "secondary", "transition", "transitions", "start", "end",
    "distractor", "distractors", "forbidden",
}
STRUCTURAL_LABELS = {
    "states", "actions", "scene", "camera", "tempo",
    "summary", "object_primary", "object_secondary", "action_chain", "end_state",
    "scene_start", "scene_end", "scene_transition", "camera_transition", "hand_interaction",
    "distractors", "forbidden", "keep", "remove", "replace_with",
    "action_chain_after", "end_state_after", "scene_start_after", "scene_end_after",
    "scene_transition_after", "camera_transition_after", "hand_interaction_after",
}


def _normalize_token_text(text: str) -> str:
    text = text.strip().lower()
    for prefix in ("▁", "Ġ", "##"):
        text = text.replace(prefix, "")
    text = text.replace("</w>", "")
    return text


def _extract_important_terms(edit_instruction: str, reasoning_trace: Optional[str]) -> Set[str]:
    text = f"{edit_instruction}\n{reasoning_trace or ''}".lower()
    pieces = re.findall(r"[a-z0-9][a-z0-9\\-_/]+", text)
    terms = set()
    for piece in pieces:
        clean = piece.strip().lower()
        if len(clean) < 3:
            continue
        if clean in STOP_WORDS or clean in STRUCTURAL_LABELS:
            continue
        terms.add(clean)
    return terms


def _is_edit_important(token_text: str, important_terms: Set[str]) -> bool:
    if not token_text or token_text in STOP_WORDS:
        return False
    if token_text in important_terms:
        return True
    if len(token_text) >= 4:
        for term in important_terms:
            if token_text in term or term in token_text:
                return True
    return False


def _important_token_positions(token_texts: List[str], important_terms: Set[str]) -> Set[int]:
    positions = set()
    for idx, text in enumerate(token_texts):
        if _is_edit_important(text, important_terms):
            positions.add(idx)
    return positions


def _token_weights(token_strings: List[str], scheme: str, cfg: Config, context: Optional[Dict[str, object]] = None) -> torch.Tensor:
    if scheme == "uniform":
        return torch.ones(len(token_strings), dtype=torch.float32)

    important_terms = set()
    late_token_ratio = 1.0
    important_positions = set()
    if context:
        important_terms = context.get("important_terms") or set()
        late_token_ratio = float(context.get("late_token_ratio", 1.0))
        important_positions = context.get("important_positions") or set()

    normalized_tokens = [_normalize_token_text(token) for token in token_strings]
    if scheme == "edit_window" and important_terms and not important_positions:
        important_positions = _important_token_positions(normalized_tokens, important_terms)

    weights = []
    late_start = int(len(token_strings) * (1.0 - late_token_ratio)) if late_token_ratio < 1.0 else 0
    for idx, text in enumerate(normalized_tokens):
        if not text:
            weight = 0.1
        elif text in STRUCTURAL_LABELS:
            weight = cfg.structural_token_weight
        elif text in STOP_WORDS:
            weight = 0.2 if scheme in {"basic", "edit_aware"} else 0.3
        elif len(text) <= 2:
            weight = 0.4
        elif text.startswith("<") and text.endswith(">"):
            weight = 0.0
        else:
            weight = 1.0

        if scheme == "edit_aware" and _is_edit_important(text, important_terms):
            weight *= cfg.edit_term_boost
        elif scheme == "edit_window":
            if idx in important_positions:
                weight *= cfg.edit_match_weight
            elif important_positions:
                nearest = min(abs(idx - p) for p in important_positions)
                if nearest <= cfg.edit_window_radius:
                    weight *= max(cfg.edit_neighbor_weight - 0.2 * nearest, 1.0)
                else:
                    weight *= cfg.edit_background_weight

        if idx >= late_start:
            weight *= 1.15

        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32)


def _pool_hidden_steps(
    hidden_steps: List[torch.Tensor],
    token_strings: Optional[List[str]],
    cfg: Config,
    weighting_scheme_override: Optional[str] = None,
    weighting_context: Optional[Dict[str, object]] = None,
) -> torch.Tensor:
    stacked = torch.stack(hidden_steps, dim=1)
    mode = (cfg.embedding_pooling or "mean").lower()
    scheme = weighting_scheme_override or cfg.weighting_scheme

    if mode == "last":
        pooled = stacked[:, -1, :]
    elif mode == "max":
        pooled = stacked.max(dim=1).values
    elif mode == "weighted_mean" and token_strings is not None:
        weights = _token_weights(token_strings, scheme, cfg, weighting_context)
        weights = weights / (weights.sum() + 1e-8)
        pooled = (stacked * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
    else:
        pooled = stacked.mean(dim=1)

    pooled = pooled[0]
    if cfg.normalize_embeddings:
        pooled = F.normalize(pooled.unsqueeze(0), dim=-1).squeeze(0)
    return pooled.detach().cpu().float()


def _hook_penultimate_hidden(_module, inputs, _output, container):
    if inputs and isinstance(inputs[0], torch.Tensor):
        hidden = inputs[0]
        container["steps"].append(hidden[:, -1, :].detach().float().cpu())


class LocalQwenRunner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None
        self.processor = None
        self.prompts = get_prompts(cfg.prompt_style)

    def load(self) -> None:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs = {
            "dtype": dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
        }
        if self.cfg.attn_implementation:
            kwargs["attn_implementation"] = self.cfg.attn_implementation

        self.model = AutoModelForImageTextToText.from_pretrained(self.cfg.model_name, **kwargs)
        self.processor = AutoProcessor.from_pretrained(self.cfg.model_name, trust_remote_code=True)
        LOGGER.info("Loaded model from %s", self.cfg.model_name)

    def _device(self):
        return next(self.model.parameters()).device

    def _move_to_device(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = self._device()
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    def _build_inputs(self, video_path: str, prompt: str) -> Dict[str, torch.Tensor]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": f"file://{os.path.abspath(video_path)}", "fps": float(self.cfg.sample_fps)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        images, videos, _ = process_vision_info(messages, return_video_kwargs=True)
        template_kwargs = {}
        if self.cfg.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.cfg.enable_thinking
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        inputs = self.processor(
            text=[text],
            images=images if images else None,
            videos=videos if videos else None,
            padding=True,
            return_tensors="pt",
        )
        return self._move_to_device(inputs)

    def _build_text_only_inputs(self, prompt: str) -> Dict[str, torch.Tensor]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        template_kwargs = {}
        if self.cfg.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.cfg.enable_thinking
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        inputs = self.processor(
            text=[text],
            images=None,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        return self._move_to_device(inputs)

    def _register_embedding_hook(self):
        container = {"steps": []}
        target = self.model.get_output_embeddings()
        handle = target.register_forward_hook(partial(_hook_penultimate_hidden, container=container))
        return container, handle

    def _generate(
        self,
        video_path: str,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        capture_embedding: bool,
        weighting_scheme_override: Optional[str] = None,
        weighting_context: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        inputs = self._build_inputs(video_path, prompt)
        container = {"steps": []}
        handle = None
        if capture_embedding:
            container, handle = self._register_embedding_hook()

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
            "use_cache": True,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs.update(
                {
                    "temperature": temperature,
                    "top_p": self.cfg.top_p,
                    "top_k": self.cfg.top_k,
                }
            )
        if self.cfg.min_p is not None:
            gen_kwargs["min_p"] = self.cfg.min_p
        if self.cfg.repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = self.cfg.repetition_penalty

        with torch.no_grad():
            if self.cfg.mixed_precision:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                with torch.autocast(device_type="cuda", dtype=dtype):
                    output_ids = self.model.generate(**inputs, **gen_kwargs)
            else:
                output_ids = self.model.generate(**inputs, **gen_kwargs)

        if handle:
            handle.remove()

        input_length = inputs["input_ids"].shape[1]
        trimmed = output_ids[:, input_length:]
        text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        tokens = trimmed[0].tolist()
        token_strings = [self.processor.tokenizer.decode([token]) for token in tokens]

        embedding = None
        if capture_embedding and container["steps"]:
            token_list = token_strings if self.cfg.embedding_pooling == "weighted_mean" else None
            embedding = _pool_hidden_steps(
                container["steps"],
                token_list,
                self.cfg,
                weighting_scheme_override=weighting_scheme_override,
                weighting_context=weighting_context,
            ).numpy()

        return {
            "text": text,
            "embedding": embedding,
            "token_mapping": {
                "tokens": tokens,
                "token_strings": token_strings,
                "full_text": text,
            },
        }

    def _generate_text_only(
        self,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ) -> Dict[str, object]:
        inputs = self._build_text_only_inputs(prompt)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
            "use_cache": True,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs.update(
                {
                    "temperature": temperature,
                    "top_p": self.cfg.top_p,
                    "top_k": self.cfg.top_k,
                }
            )
        if self.cfg.min_p is not None:
            gen_kwargs["min_p"] = self.cfg.min_p
        if self.cfg.repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = self.cfg.repetition_penalty

        with torch.no_grad():
            if self.cfg.mixed_precision:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                with torch.autocast(device_type="cuda", dtype=dtype):
                    output_ids = self.model.generate(**inputs, **gen_kwargs)
            else:
                output_ids = self.model.generate(**inputs, **gen_kwargs)

        input_length = inputs["input_ids"].shape[1]
        trimmed = output_ids[:, input_length:]
        text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        tokens = trimmed[0].tolist()
        token_strings = [self.processor.tokenizer.decode([token]) for token in tokens]
        return {
            "text": text,
            "embedding": None,
            "token_mapping": {
                "tokens": tokens,
                "token_strings": token_strings,
                "full_text": text,
            },
        }

    def embed_reference(self, video_path: str) -> Dict[str, object]:
        return self._generate(
            video_path=video_path,
            prompt=self.prompts["reference"](),
            max_new_tokens=self.cfg.max_new_tokens_reference,
            do_sample=self.cfg.do_sample_reference,
            temperature=self.cfg.temperature_reference,
            capture_embedding=True,
            weighting_scheme_override=self.cfg.gallery_weighting_scheme,
        )

    def generate_reasoning(self, video_path: str, edit_instruction: str) -> Dict[str, object]:
        traces = []
        trace_count = 1
        if self.cfg.reasoning_strategy == "self_consistency":
            trace_count = max(self.cfg.self_consistency_samples, 1)
        elif self.cfg.reasoning_samples > 1:
            trace_count = self.cfg.reasoning_samples

        for _ in range(trace_count):
            result = self._generate(
                video_path=video_path,
                prompt=self.prompts["edit_reasoning"](edit_instruction),
                max_new_tokens=self.cfg.max_new_tokens_reasoning,
                do_sample=self.cfg.do_sample_reasoning,
                temperature=self.cfg.temperature_reasoning,
                capture_embedding=False,
            )
            text = result["text"].strip()
            if text:
                traces.append(text)

        if not traces:
            raise RuntimeError(f"Failed to generate reasoning trace for {video_path}")

        final_trace = traces[0]
        if len(traces) > 1:
            synthesis = self._generate(
                video_path=video_path,
                prompt=self.prompts["consistency_synthesis"](edit_instruction, traces),
                max_new_tokens=self.cfg.max_new_tokens_reasoning,
                do_sample=self.cfg.do_sample_reasoning,
                temperature=self.cfg.temperature_synthesis,
                capture_embedding=False,
            )["text"].strip()
            if synthesis:
                final_trace = synthesis

        return {"reasoning": final_trace, "all_traces": traces}

    def embed_edited_query(self, video_path: str, edit_instruction: str, reasoning_trace: str) -> Dict[str, object]:
        important_terms = _extract_important_terms(edit_instruction, reasoning_trace)
        return self._generate(
            video_path=video_path,
            prompt=self.prompts["edit_description"](edit_instruction, reasoning_trace),
            max_new_tokens=self.cfg.max_new_tokens_description,
            do_sample=self.cfg.do_sample_description,
            temperature=self.cfg.temperature_description,
            capture_embedding=True,
            weighting_scheme_override=self.cfg.query_weighting_scheme,
            weighting_context={
                "important_terms": important_terms,
                "late_token_ratio": self.cfg.late_token_ratio,
            },
        )

    def embed_single_stage_query(self, video_path: str, edit_instruction: str) -> Dict[str, object]:
        important_terms = _extract_important_terms(edit_instruction, None)
        return self._generate(
            video_path=video_path,
            prompt=self.prompts["edit"](edit_instruction),
            max_new_tokens=self.cfg.max_new_tokens_description,
            do_sample=self.cfg.do_sample_description,
            temperature=self.cfg.temperature_description,
            capture_embedding=True,
            weighting_scheme_override=self.cfg.query_weighting_scheme,
            weighting_context={
                "important_terms": important_terms,
                "late_token_ratio": self.cfg.late_token_ratio,
            },
        )

    def generate_text(
        self,
        video_path: str,
        prompt: str,
        max_new_tokens: int = 192,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> Dict[str, object]:
        return self._generate(
            video_path=video_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            capture_embedding=False,
        )

    def generate_text_only(
        self,
        prompt: str,
        max_new_tokens: int = 192,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> Dict[str, object]:
        return self._generate_text_only(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )


def write_jsonl(path: str | Path, rows: List[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
