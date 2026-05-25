from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


COVR_DATA_ROOT = Path(os.environ.get("COVR_DATA_ROOT", "/data_1/ldq/data/CoVR-R"))
WEBVID_ROOT = Path(os.environ.get("COVR_WEBVID_ROOT", COVR_DATA_ROOT / "WebVid/8M/train"))
SS2_ROOT = Path(os.environ.get("COVR_SS2_ROOT", COVR_DATA_ROOT / "something_something_v2/20bn-something-something-v2"))
DEFAULT_LABEL_PATH = Path(os.environ.get("COVR_LABEL_PATH", COVR_DATA_ROOT / "test-set_no-labels.json"))


@dataclass(frozen=True)
class QueryRecord:
    split: str
    query_id: int
    video_source: str | int
    modification_text: str
    video_path: str


def resolve_video_path(split: str, video_source: str | int) -> str:
    if split == "webvid":
        return str(WEBVID_ROOT / f"{video_source}.mp4")
    if split == "ss2":
        return str(SS2_ROOT / f"{video_source}.webm")
    raise ValueError(f"Unsupported split: {split}")


def load_queries(label_path: str | Path = DEFAULT_LABEL_PATH) -> Dict[str, List[QueryRecord]]:
    data = json.loads(Path(label_path).read_text())
    output: Dict[str, List[QueryRecord]] = {"webvid": [], "ss2": []}

    for section in data:
        for split, rows in section.items():
            if split not in output:
                raise ValueError(f"Unexpected split in label file: {split}")
            for row in rows:
                output[split].append(
                    QueryRecord(
                        split=split,
                        query_id=int(row["id"]),
                        video_source=row["video_source"],
                        modification_text=row["modification_text"],
                        video_path=resolve_video_path(split, row["video_source"]),
                    )
                )

    return output


def list_gallery_ids(split: str) -> List[str]:
    if split == "webvid":
        return sorted(
            _strip_suffix(str(path.relative_to(WEBVID_ROOT)), ".mp4")
            for path in WEBVID_ROOT.rglob("*.mp4")
        )
    if split == "ss2":
        return sorted(path.stem for path in SS2_ROOT.glob("*.webm"))
    raise ValueError(f"Unsupported split: {split}")


def validate_query_paths(records: Iterable[QueryRecord]) -> List[str]:
    missing = []
    for record in records:
        if not Path(record.video_path).exists():
            missing.append(record.video_path)
    return missing


def load_id_list(path: str | Path) -> List[str]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(line)
    return rows


def load_query_key_list(path: str | Path) -> set[tuple[int, str]]:
    keys = set()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"Expected query key line as <query_id><tab><video_source>, got: {line}")
        keys.add((int(parts[0]), parts[1]))
    return keys


def _strip_suffix(text: str, suffix: str) -> str:
    if text.endswith(suffix):
        return text[: -len(suffix)]
    return text
