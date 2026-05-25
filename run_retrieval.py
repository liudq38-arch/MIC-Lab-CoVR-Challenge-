from __future__ import annotations

import json
import logging
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from config import Config, load_config_from_args
from covr_data import QueryRecord, load_queries, load_query_key_list
from model_runner import LocalQwenRunner, write_jsonl


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def _splits(name: str):
    return ["webvid", "ss2"] if name == "all" else [name]


def _load_gallery(cfg: Config, split: str):
    npz_path = Path(cfg.gallery_dir) / f"{split}_gallery_embeddings.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing gallery embeddings: {npz_path}")
    with np.load(npz_path) as data:
        candidate_ids = sorted(data.files)
        matrix = np.stack([np.asarray(data[candidate_id], dtype=np.float32) for candidate_id in candidate_ids])
    matrix = torch.from_numpy(matrix).float()
    matrix = F.normalize(matrix, dim=-1) if cfg.normalize_embeddings else matrix
    return candidate_ids, matrix


def _top_predictions(scores: np.ndarray, candidate_ids, k: int):
    order = np.argsort(scores)[::-1][:k]
    return [candidate_ids[index] for index in order]


def _safe_source_key(video_source) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", str(video_source))


def _cache_path(cfg: Config, split: str, record: QueryRecord) -> Path:
    return Path(cfg.query_cache_dir) / split / f"{record.query_id}__{_safe_source_key(record.video_source)}.json"


def _load_cached_query(cfg: Config, split: str, record: QueryRecord):
    if not cfg.reuse_query_cache:
        return None
    path = _cache_path(cfg, split, record)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    embedding = np.asarray(payload["embedding"], dtype=np.float32)
    payload["embedding"] = embedding
    return payload


def _save_cached_query(cfg: Config, split: str, record: QueryRecord, payload: dict):
    if not cfg.save_query_cache:
        return
    path = _cache_path(cfg, split, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["embedding"] = np.asarray(data["embedding"], dtype=np.float32).tolist()
    path.write_text(json.dumps(data, ensure_ascii=False))


def _save_query_embedding(cfg: Config, split: str, record: QueryRecord, embedding: np.ndarray):
    path = Path(cfg.query_embedding_dir) / split / f"{record.query_id}__{_safe_source_key(record.video_source)}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(embedding, dtype=np.float32))


def _submission_entry(cfg: Config, record: QueryRecord, predictions, reasoning_trace: str):
    entry = {
        "id": record.query_id,
        "video_source": record.video_source,
        "video_target": predictions,
    }
    if cfg.submission_mode == "test":
        entry["reasoning_trace"] = [reasoning_trace]
    return entry


def main() -> None:
    cfg = load_config_from_args()
    queries = load_queries(cfg.label_path) if cfg.label_path else load_queries()
    runner = LocalQwenRunner(cfg)
    runner.load()

    submission = []
    debug_rows = []
    score_rows = []

    for split in _splits(cfg.split):
        candidate_ids, gallery_matrix = _load_gallery(cfg, split)
        candidate_index = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
        records = queries[split]
        if cfg.query_keys_path:
            allow_keys = load_query_key_list(cfg.query_keys_path)
            records = [record for record in records if (int(record.query_id), str(record.video_source)) in allow_keys]
        if cfg.query_offset:
            records = records[cfg.query_offset :]
        if cfg.limit:
            records = records[: cfg.limit]

        block = []
        progress = tqdm(
            records,
            desc=f"{split} queries",
            dynamic_ncols=True,
        )
        for index, record in enumerate(progress, start=1):
            try:
                cached = _load_cached_query(cfg, split, record)
                if cached is not None:
                    reasoning = {
                        "reasoning": cached.get("reasoning"),
                        "all_traces": cached.get("all_reasoning_traces"),
                    }
                    edited = {
                        "text": cached["edited_description"],
                        "embedding": cached["embedding"],
                        "token_mapping": cached.get("token_mapping"),
                    }
                else:
                    if cfg.reasoning_strategy == "single_stage":
                        reasoning = {"reasoning": None, "all_traces": None}
                        edited = runner.embed_single_stage_query(record.video_path, record.modification_text)
                    else:
                        reasoning = runner.generate_reasoning(record.video_path, record.modification_text)
                        edited = runner.embed_edited_query(record.video_path, record.modification_text, reasoning["reasoning"])
                if edited["embedding"] is None:
                    raise RuntimeError(f"No query embedding produced for {split}:{record.query_id}")

                query_embedding = torch.from_numpy(np.asarray(edited["embedding"], dtype=np.float32)).float()
                query_embedding = F.normalize(query_embedding.unsqueeze(0), dim=-1).squeeze(0) if cfg.normalize_embeddings else query_embedding
                scores = torch.matmul(gallery_matrix, query_embedding).cpu().numpy()

                source_key = str(record.video_source)
                if cfg.exclude_reference and source_key in candidate_index:
                    scores[candidate_index[source_key]] = -np.inf

                predictions = _top_predictions(scores, candidate_ids, cfg.top_k_predictions)
                top_score_order = np.argsort(scores)[::-1][: cfg.score_dump_top_k]
                block.append(_submission_entry(cfg, record, predictions, reasoning["reasoning"]))
                debug_rows.append(
                    {
                        "split": split,
                        "query_id": record.query_id,
                        "video_source": record.video_source,
                        "video_path": record.video_path,
                        "modification_text": record.modification_text,
                        "reasoning": reasoning["reasoning"],
                        "all_reasoning_traces": reasoning["all_traces"],
                        "edited_description": edited["text"],
                        "predictions_top5": predictions[:5],
                        "token_mapping": edited["token_mapping"],
                    }
                )
                _save_query_embedding(cfg, split, record, edited["embedding"])
                score_rows.append(
                    {
                        "split": split,
                        "query_id": record.query_id,
                        "video_source": record.video_source,
                        "top_candidates": [candidate_ids[i] for i in top_score_order],
                        "top_scores": [float(scores[i]) for i in top_score_order],
                    }
                )
                _save_cached_query(
                    cfg,
                    split,
                    record,
                    {
                        "split": split,
                        "query_id": record.query_id,
                        "video_source": record.video_source,
                        "video_path": record.video_path,
                        "modification_text": record.modification_text,
                        "reasoning": reasoning["reasoning"],
                        "all_reasoning_traces": reasoning["all_traces"],
                        "edited_description": edited["text"],
                        "token_mapping": edited["token_mapping"],
                        "embedding": edited["embedding"],
                    },
                )
                del query_embedding
                del scores
                del top_score_order
                progress.set_postfix(done=index, refresh=False)
                if index % 50 == 0:
                    LOGGER.info("%s processed %d/%d queries", split, index, len(records))
            except Exception as exc:
                LOGGER.error("Failed on %s query_id=%s source=%s: %s", split, record.query_id, record.video_source, exc)
            finally:
                torch.cuda.empty_cache()

        submission.append({split: block})

    submission_path = Path(cfg.submission_path)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text(json.dumps(submission, ensure_ascii=False, indent=2))
    write_jsonl(cfg.query_debug_path, debug_rows)
    if cfg.score_dump_path:
        write_jsonl(cfg.score_dump_path, score_rows)

    LOGGER.info("Saved submission to %s", submission_path)
    LOGGER.info("Saved debug rows to %s", cfg.query_debug_path)
    if cfg.score_dump_path:
        LOGGER.info("Saved score dump to %s", cfg.score_dump_path)


if __name__ == "__main__":
    main()
