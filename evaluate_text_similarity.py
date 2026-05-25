from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def load_jsonl(path: str | Path) -> List[dict]:
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def build_gallery(gallery_jsonl: str | Path):
    rows = load_jsonl(gallery_jsonl)
    candidate_ids = [str(row["candidate_id"]) for row in rows]
    texts = [row["reference_text"] for row in rows]
    return candidate_ids, texts


def build_queries(query_jsonl: str | Path, split: str):
    rows = load_jsonl(query_jsonl)
    if rows and "split" in rows[0]:
        rows = [row for row in rows if row.get("split") == split]
    query_ids = [int(row["query_id"]) for row in rows]
    sources = [str(row["video_source"]) for row in rows]
    texts = [row["edited_description"] for row in rows]
    return rows, query_ids, sources, texts


def make_submission(split: str, rows: List[dict], predictions: List[List[str]], output_path: str | Path):
    payload = [{split: []}]
    for row, preds in zip(rows, predictions):
        payload[0][split].append(
            {
                "id": int(row["query_id"]),
                "video_source": row["video_source"],
                "video_target": preds,
            }
        )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return path


def main():
    parser = argparse.ArgumentParser(description="Run a TF-IDF text similarity retrieval baseline.")
    parser.add_argument("--split", required=True, choices=["webvid", "ss2"])
    parser.add_argument("--gallery-jsonl", required=True)
    parser.add_argument("--query-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--score-dump", default="", help="Optional jsonl path for top candidate raw text-sim scores.")
    parser.add_argument("--score-dump-top-k", type=int, default=300)
    args = parser.parse_args()

    candidate_ids, gallery_texts = build_gallery(args.gallery_jsonl)
    rows, _query_ids, sources, query_texts = build_queries(args.query_jsonl, args.split)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    gallery_matrix = vectorizer.fit_transform(gallery_texts)
    query_matrix = vectorizer.transform(query_texts)

    scores = linear_kernel(query_matrix, gallery_matrix)
    candidate_index = {candidate_id: idx for idx, candidate_id in enumerate(candidate_ids)}

    predictions: List[List[str]] = []
    score_rows = []
    for row_idx, source in enumerate(sources):
        if source in candidate_index:
            scores[row_idx, candidate_index[source]] = -np.inf
        score_order = np.argsort(scores[row_idx])[::-1][: args.score_dump_top_k]
        order = score_order[: args.top_k]
        predictions.append([candidate_ids[i] for i in order])
        if args.score_dump:
            score_rows.append(
                {
                    "split": args.split,
                    "query_id": int(rows[row_idx]["query_id"]),
                    "video_source": rows[row_idx]["video_source"],
                    "top_candidates": [candidate_ids[i] for i in score_order],
                    "top_scores": [float(scores[row_idx][i]) for i in score_order],
                }
            )

    output_path = make_submission(args.split, rows, predictions, args.output)
    if args.score_dump:
        dump_path = Path(args.score_dump)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open("w", encoding="utf-8") as handle:
            for row in score_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(output_path)


if __name__ == "__main__":
    main()
