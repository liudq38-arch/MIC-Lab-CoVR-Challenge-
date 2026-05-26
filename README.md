# MIC-Lab CoVR-R Challenge

This repository contains the minimal code for our CoVR-R challenge method.

## Method

The system is a zero-shot reason-then-retrieve pipeline:

1. generate retrieval-oriented descriptions for gallery videos,
2. infer structured edit reasoning from the reference video and modification text,
3. generate an edited target-video description,
4. pool generated-token hidden states into dense embeddings,
5. retrieve by cosine similarity,
6. run a TF-IDF text branch over generated descriptions,
7. fuse dense and sparse scores.

## Contents

```text
.
├── configs/                       # retrieval configs
├── config.py
├── covr_data.py
├── model_runner.py
├── prompts.py
├── generate_gallery_embeddings.py
├── run_retrieval.py
├── evaluate_text_similarity.py
└── apply_score_fusion.py
```

Large generated artifacts, logs, datasets, model weights, and prediction JSON files are intentionally not included.

## Environment

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The code expects a local Qwen3.5-27B-compatible video-language model path.
Set `model_name` in the YAML config to your local model path.

## Data Paths

The loader uses these environment variables when available:

```bash
export COVR_DATA_ROOT=/path/to/CoVR-R
export COVR_WEBVID_ROOT=$COVR_DATA_ROOT/WebVid/8M/train
export COVR_SS2_ROOT=$COVR_DATA_ROOT/something_something_v2/20bn-something-something-v2
export COVR_LABEL_PATH=$COVR_DATA_ROOT/test-set_no-labels.json
```

You can also edit the defaults in `covr_data.py`.

## Main Config

```text
configs/qwen35_nothink_strict_mainline_test.yaml
```

Additional prompt variants used during method development:

```text
configs/qwen35_nothink_transition_strict_ss2.yaml
configs/qwen35_nothink_exact_slots_gallery_ss2.yaml
```

## Example Commands

Generate gallery embeddings:

```bash
python3 generate_gallery_embeddings.py \
  --config configs/qwen35_nothink_strict_mainline_test.yaml \
  --split webvid

python3 generate_gallery_embeddings.py \
  --config configs/qwen35_nothink_strict_mainline_test.yaml \
  --split ss2
```

Run dense retrieval:

```bash
python3 run_retrieval.py \
  --config configs/qwen35_nothink_strict_mainline_test.yaml \
  --split webvid \
  --submission-mode test

python3 run_retrieval.py \
  --config configs/qwen35_nothink_strict_mainline_test.yaml \
  --split ss2 \
  --submission-mode test
```

Run TF-IDF retrieval after gallery text and query debug files have been produced:

```bash
python3 evaluate_text_similarity.py \
  --split webvid \
  --gallery-jsonl <webvid_reference_text.jsonl> \
  --query-jsonl <webvid_query_debug.jsonl> \
  --output <webvid_tfidf_submission.json> \
  --score-dump <webvid_tfidf_scores.jsonl>
```

Fuse dense and sparse scores:

```bash
python3 apply_score_fusion.py \
  --embedding <embedding_scores.jsonl> \
  --tfidf <tfidf_scores.jsonl> \
  --split webvid \
  --mode minmax \
  --alpha 0.8 \
  --output <webvid_fused.json>
```

Repeat the same commands with `--split ss2` for the SS2 branch.
