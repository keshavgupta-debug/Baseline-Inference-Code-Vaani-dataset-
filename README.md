# STIPA Project - Baseline ASR Evaluation Package

This directory contains the complete codebase, configuration, and tools to run and analyze the baseline ASR benchmark on the STIPA (Vaani) dataset using AI4Bharat's `indicconformer_stt_hi_hybrid_ctc_rnnt_large`.

---

## Prerequisites

Before running evaluation, you need:

| Requirement | Details |
| :--- | :--- |
| **Python 3.10** | Required for AI4Bharat NeMo compatibility. |
| **CUDA GPU** | Inference loads the Conformer on GPU (`--gpu` selects device index). |
| **GCS access** | Audio lives at `gs://karya-language-server-main/hindi-stipa/...`. Provide a service account JSON via `--key` (e.g. `stipa_key.json`). |
| **Hugging Face** | Gated model [indicconformer_stt_hi_hybrid_ctc_rnnt_large](https://huggingface.co/ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large) — request access and log in before first run. |
| **AI4Bharat NeMo** | Not installed by `requirements.txt` alone; clone and install the fork (see setup below). |

Verify NeMo after install:

```bash
python -c "import nemo.collections.asr as nemo_asr; print('NeMo OK')"
```

---

## What is not included in this folder

Obtain these separately from the STIPA / Vaani dataset owners or whoever ran the original benchmark:

| Artifact | Purpose |
| :--- | :--- |
| **`stipa_key.json`** | GCS service account key with read access to the STIPA audio bucket. |
| **`final_test.jsonl`** | Official test manifest (~33,244 utterances, ~51.5 h). Required for full benchmark re-run. |
| **Other split manifests** | `final_train_all.jsonl`, `final_train_script.jsonl`, `final_val.jsonl` — only needed for split analysis scripts. |
| **Result CSVs** | Optional reference outputs: `baseline_summary_by_language.csv`, `baseline_predictions_detailed.csv` (produced by `run_baseline_eval.py` or shared from a completed run). |

See [HANDOFF.md](HANDOFF.md) for a short checklist when sharing this package with a colleague.

---

## Manifest schema (JSONL)

Each line is one JSON object. Example:

```json
{"audio_filepath": "gs://karya-language-server-main/hindi-stipa/hindi/hindi_0001.wav", "duration": 2.5, "text": "reference transcript", "language": "hindi", "variety": "hindi", "gender": "Male"}
```

| Field | Description |
| :--- | :--- |
| `audio_filepath` | `gs://` URI to the WAV file |
| `duration` | Length in seconds (float) |
| `text` | Reference transcript for WER/CER |
| `language` | Language / variety label (lowercase in metrics) |
| `variety` | Dialect label (often same as `language`) |
| `gender` | e.g. `Male`, `Female` |

---

## Files included

| File | Description |
| :--- | :--- |
| **`run_baseline_eval.py`** | **Main evaluation engine.** Downloads audio from GCS, loads the Conformer on GPU, batched inference, text normalization, WER/CER, CSV export. |
| **`smoke_test.jsonl`** | 34 clips (shortest + longest per language). Rapid env check (~1–2 min including downloads). |
| **`create_34_smoke_test.py`** | Regenerate `smoke_test.jsonl` from a full test manifest. |
| **`correlate_splits.py`** | Cross-split correlation vs baseline WER (optional; needs split manifests + summary CSV). |
| **`generate_split_tables.py`** | Markdown distribution tables per split (optional). |
| **`analyze_all_splits.py`** | Profiles utterance counts, hours, gender per split (optional). |
| **`calc_metrics.py`** | Recompute WER/CER from `baseline_predictions_detailed.csv`. |
| **`requirements.txt`** | Pinned torch / lightning / numpy stack for NeMo. |
| **`HANDOFF.md`** | Checklist of artifacts to send separately (GCS key, full manifest, optional reference CSVs). |
| **`stipa_asr_baseline_evaluation_report.md`** | Benchmark report and per-dialect analysis. |

---

## Quickstart guide

### 1. Environment setup (Python 3.10)

```bash
conda create -y -n stipa_env python=3.10
conda activate stipa_env

pip install -r requirements.txt

git clone --branch nemo-v2 https://github.com/AI4Bharat/NeMo.git
cd NeMo && pip install -e ".[asr]" && cd ..

python -c "import nemo.collections.asr as nemo_asr; print('NeMo OK')"
```

**Hugging Face login**

Linux / macOS:

```bash
export HF_TOKEN="your_hf_read_token"
python -c "from huggingface_hub import login; login('$HF_TOKEN')"
```

Windows PowerShell:

```powershell
$env:HF_TOKEN = "your_hf_read_token"
python -c "from huggingface_hub import login; import os; login(os.environ['HF_TOKEN'])"
```

Or: `huggingface-cli login`

Place `stipa_key.json` in this directory (or pass an absolute path to `--key`).

### 2. Smoke test (~34 clips)

Success criteria: script finishes without errors, prints a per-language summary table, and writes CSVs under `--output_dir`. Metrics will differ slightly from the full test; the goal is to validate GCS, HF, NeMo, and GPU.

```bash
python run_baseline_eval.py \
  --manifest smoke_test.jsonl \
  --key stipa_key.json \
  --batch_size 16 \
  --decoder rnnt \
  --gpu 0
```

### 3. Full test set (~51.5 hours of audio)

Requires `final_test.jsonl` (not shipped in this repo).

```bash
python run_baseline_eval.py \
  --manifest final_test.jsonl \
  --key stipa_key.json \
  --batch_size 32 \
  --decoder rnnt \
  --gpu 0 \
  --output_dir ./stipa_full_eval
```

Expected headline metrics (reference run, RNNT decoder): overall **~33.01% WER / ~15.40% CER**; see the evaluation report for per-language targets.

### 4. Optional split analysis

Put manifests under `./data/` (or pass paths explicitly):

```bash
python analyze_all_splits.py --data-dir ./data --baseline-summary ./stipa_full_eval/baseline_summary_by_language.csv
python correlate_splits.py --data-dir ./data --baseline-summary ./stipa_full_eval/baseline_summary_by_language.csv
python generate_split_tables.py --data-dir ./data
python calc_metrics.py --predictions ./stipa_full_eval/baseline_predictions_detailed.csv
```

---

## Summary of baseline benchmark results

* **Standard Hindi (35.94h):** **22.76% WER | 10.46% CER**
* **Regional varieties (15.52h, 16 dialects):** **59.66% WER | 27.69% CER**
* **Overall test set (51.46h):** **33.01% WER | 15.40% CER**
