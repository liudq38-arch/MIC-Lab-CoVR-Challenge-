from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from config import load_config_from_args
from covr_data import list_gallery_ids, load_id_list, resolve_video_path
from model_runner import LocalQwenRunner, write_jsonl


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def _splits(name: str):
    return ["webvid", "ss2"] if name == "all" else [name]


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _atomic_write_text(path: Path, content: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content)
    tmp_path.replace(path)


def _save_gallery_state(
    split: str,
    gallery_dir: Path,
    gallery_text_log_dir: Path,
    embeddings: dict,
    text_rows: list,
) -> None:
    npz_path = gallery_dir / f"{split}_gallery_embeddings.npz"
    index_path = gallery_dir / f"{split}_gallery_index.json"
    text_log_path = gallery_text_log_dir / f"{split}_reference_text.jsonl"

    npz_tmp = npz_path.with_suffix(npz_path.suffix + ".tmp")
    np.savez_compressed(npz_tmp, **embeddings)
    if not npz_tmp.exists() and npz_tmp.with_suffix(npz_tmp.suffix + ".npz").exists():
        npz_tmp = npz_tmp.with_suffix(npz_tmp.suffix + ".npz")
    npz_tmp.replace(npz_path)
    _atomic_write_text(
        index_path,
        json.dumps({"split": split, "candidate_ids": sorted(embeddings.keys())}, ensure_ascii=False, indent=2),
    )
    write_jsonl(text_log_path, text_rows)


def _load_gallery_state(split: str, gallery_dir: Path, gallery_text_log_dir: Path):
    npz_path = gallery_dir / f"{split}_gallery_embeddings.npz"
    text_log_path = gallery_text_log_dir / f"{split}_reference_text.jsonl"
    embeddings = {}
    if npz_path.exists():
        with np.load(npz_path) as data:
            for key in data.files:
                embeddings[key] = np.asarray(data[key], dtype=np.float32)
    text_rows = _read_jsonl(text_log_path)
    return embeddings, text_rows


def main() -> None:
    cfg = load_config_from_args()
    runner = LocalQwenRunner(cfg)
    runner.load()

    gallery_dir = Path(cfg.gallery_dir)
    gallery_text_log_dir = Path(cfg.gallery_text_log_dir)
    gallery_dir.mkdir(parents=True, exist_ok=True)
    gallery_text_log_dir.mkdir(parents=True, exist_ok=True)

    for split in _splits(cfg.split):
        gallery_ids = list_gallery_ids(split)
        if cfg.candidate_ids_path:
            allow = set(load_id_list(cfg.candidate_ids_path))
            gallery_ids = [candidate_id for candidate_id in gallery_ids if candidate_id in allow]
        if cfg.gallery_offset:
            gallery_ids = gallery_ids[cfg.gallery_offset :]
        if cfg.limit:
            gallery_ids = gallery_ids[: cfg.limit]

        embeddings, text_rows = ({}, [])
        if cfg.resume_gallery:
            embeddings, text_rows = _load_gallery_state(split, gallery_dir, gallery_text_log_dir)
            if embeddings:
                LOGGER.info("Resuming %s gallery from %d saved embeddings", split, len(embeddings))

        processed_ids = set(embeddings.keys())
        pending_ids = [candidate_id for candidate_id in gallery_ids if candidate_id not in processed_ids]
        new_rows = 0

        progress = tqdm(total=len(gallery_ids), initial=len(processed_ids), desc=f"{split} gallery", dynamic_ncols=True)
        for candidate_id in pending_ids:
            video_path = resolve_video_path(split, candidate_id)
            result = runner.embed_reference(video_path)
            if result["embedding"] is None:
                LOGGER.warning("No embedding for %s:%s", split, candidate_id)
                continue
            embeddings[candidate_id] = np.asarray(result["embedding"], dtype=np.float32)
            text_rows.append(
                {
                    "split": split,
                    "candidate_id": candidate_id,
                    "video_path": video_path,
                    "reference_text": result["text"],
                    "token_mapping": result["token_mapping"],
                }
            )
            processed_ids.add(candidate_id)
            new_rows += 1
            progress.update(1)
            progress.set_postfix(done=len(processed_ids), refresh=False)
            if len(processed_ids) % 100 == 0:
                LOGGER.info("%s processed %d/%d gallery videos", split, len(processed_ids), len(gallery_ids))
            if new_rows >= cfg.gallery_checkpoint_every:
                _save_gallery_state(split, gallery_dir, gallery_text_log_dir, embeddings, text_rows)
                LOGGER.info("%s checkpoint saved at %d/%d", split, len(processed_ids), len(gallery_ids))
                new_rows = 0

        _save_gallery_state(split, gallery_dir, gallery_text_log_dir, embeddings, text_rows)
        npz_path = gallery_dir / f"{split}_gallery_embeddings.npz"
        LOGGER.info("Saved %s embeddings: %s", split, npz_path)


if __name__ == "__main__":
    main()
