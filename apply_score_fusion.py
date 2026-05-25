from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_score_dump(path: str | Path) -> Dict[Tuple[int, str], dict]:
    rows = {}
    with Path(path).open() as handle:
        for line in handle:
            obj = json.loads(line)
            key = (int(obj["query_id"]), str(obj["video_source"]))
            rows[key] = obj
    return rows


def normalize_scores(values: List[float], mode: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if mode == "zscore":
        std = arr.std()
        if std < 1e-8:
            return np.zeros_like(arr)
        return (arr - arr.mean()) / std
    if mode == "minmax":
        lo = arr.min()
        hi = arr.max()
        if hi - lo < 1e-8:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)
    raise ValueError(mode)


def fuse_scores(emb_row: dict, tfidf_row: dict, mode: str, alpha: float, top_k: int) -> List[str]:
    all_ids = list(dict.fromkeys([str(x) for x in emb_row["top_candidates"]] + [str(x) for x in tfidf_row["top_candidates"]]))
    emb_scores_map = {str(cid): float(score) for cid, score in zip(emb_row["top_candidates"], emb_row["top_scores"])}
    tf_scores_map = {str(cid): float(score) for cid, score in zip(tfidf_row["top_candidates"], tfidf_row["top_scores"])}

    emb_scores = [emb_scores_map.get(cid, min(emb_scores_map.values()) - 1.0) for cid in all_ids]
    tf_scores = [tf_scores_map.get(cid, min(tf_scores_map.values()) - 1.0) for cid in all_ids]

    emb_norm = normalize_scores(emb_scores, mode)
    tf_norm = normalize_scores(tf_scores, mode)
    fused = alpha * emb_norm + (1.0 - alpha) * tf_norm
    order = np.argsort(fused)[::-1]
    return [all_ids[i] for i in order[:top_k]]


def main():
    parser = argparse.ArgumentParser(description="Apply fixed raw score fusion between embedding and TF-IDF dumps.")
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--tfidf", required=True)
    parser.add_argument("--split", required=True, choices=["webvid", "ss2"])
    parser.add_argument("--mode", required=True, choices=["zscore", "minmax"])
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    emb = load_score_dump(args.embedding)
    tfidf = load_score_dump(args.tfidf)
    keys = sorted(set(emb.keys()) & set(tfidf.keys()))

    payload = [{args.split: []}]
    for qid, source in keys:
        ranking = fuse_scores(emb[(qid, source)], tfidf[(qid, source)], args.mode, args.alpha, args.top_k)
        payload[0][args.split].append(
            {
                "id": qid,
                "video_source": int(source) if source.isdigit() else source,
                "video_target": ranking,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(output)


if __name__ == "__main__":
    main()
