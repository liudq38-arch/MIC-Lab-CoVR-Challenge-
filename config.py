from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Config:
    model_name: str = "/data_1/ldq/models/Qwen3.5-27B"
    attn_implementation: Optional[str] = "sdpa"
    mixed_precision: bool = True
    enable_thinking: Optional[bool] = None

    prompt_style: str = "reasoning"
    reasoning_strategy: str = "two_stage"
    sample_fps: float = 1.0
    max_new_tokens_reference: int = 96
    max_new_tokens_reasoning: int = 96
    max_new_tokens_description: int = 160

    do_sample_reference: bool = True
    do_sample_reasoning: bool = True
    do_sample_description: bool = True
    temperature_reference: float = 1.0
    temperature_reasoning: float = 1.0
    temperature_description: float = 1.0
    temperature_synthesis: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0
    reasoning_samples: int = 1
    self_consistency_samples: int = 1

    embedding_pooling: str = "weighted_mean"
    weighting_scheme: str = "basic"
    normalize_embeddings: bool = True
    query_weighting_scheme: str = "basic"
    gallery_weighting_scheme: str = "basic"
    edit_term_boost: float = 3.0
    late_token_ratio: float = 1.0
    structural_token_weight: float = 0.05
    edit_window_radius: int = 4
    edit_match_weight: float = 4.0
    edit_neighbor_weight: float = 1.8
    edit_background_weight: float = 0.2

    split: str = "all"
    label_path: str = ""
    limit: Optional[int] = None
    query_offset: int = 0
    gallery_offset: int = 0
    exclude_reference: bool = True
    top_k_predictions: int = 50
    submission_mode: str = "dev"

    artifact_dir: str = "artifacts"
    gallery_dir: str = "artifacts/gallery"
    submission_path: str = "artifacts/submission_dev.json"
    query_debug_path: str = "artifacts/query_debug.jsonl"
    gallery_text_log_dir: str = "artifacts/gallery_text"
    gallery_checkpoint_every: int = 50
    resume_gallery: bool = True
    query_cache_dir: str = "artifacts/query_cache"
    save_query_cache: bool = True
    reuse_query_cache: bool = True
    query_embedding_dir: str = "artifacts/query_embeddings"
    score_dump_path: str = ""
    score_dump_top_k: int = 200
    candidate_ids_path: str = ""
    query_keys_path: str = ""


def load_config(yaml_path: str | Path) -> Config:
    data = yaml.safe_load(Path(yaml_path).read_text()) or {}
    cfg = Config()
    for key, value in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _convert_value(field_name: str, raw: str, cfg: Config):
    field_map = {f.name: f for f in fields(Config)}
    field = field_map[field_name]
    field_type = str(field.type)
    if raw.startswith("[") or raw.startswith("{") or raw.startswith("("):
        try:
            return ast.literal_eval(raw)
        except Exception:
            pass
    if field.type is bool or "bool" in field_type:
        return raw.lower() in {"1", "true", "yes", "y"}
    if field.type is int or "int" in field_type:
        return None if raw.lower() == "none" else int(raw)
    if field.type is float or "float" in field_type:
        return float(raw)
    return raw


def load_config_from_args(argv=None) -> Config:
    parser = argparse.ArgumentParser(description="CoVR-R local retrieval pipeline")
    parser.add_argument("--config", default="", help="Optional YAML config path")
    args, extra = parser.parse_known_args(argv)

    cfg = load_config(args.config) if args.config else Config()

    index = 0
    while index < len(extra):
        arg = extra[index]
        if not arg.startswith("--"):
            index += 1
            continue
        key = arg[2:].replace("-", "_")
        if not hasattr(cfg, key):
            index += 1
            continue
        if index + 1 < len(extra) and not extra[index + 1].startswith("--"):
            raw = extra[index + 1]
            index += 2
        else:
            raw = "true"
            index += 1
        setattr(cfg, key, _convert_value(key, raw, cfg))
    return cfg
