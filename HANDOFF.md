# Colleague handoff checklist

Use this when sharing the **Baseline Model** folder so someone can **re-run baseline inference**.

## Send with the folder

1. This repository / zip (includes `run_baseline_eval.py`, `smoke_test.jsonl`, docs).
2. **`stipa_key.json`** (or instructions to obtain a GCS service account with read access to `gs://karya-language-server-main/hindi-stipa/`). **Do not commit keys to git.**
3. **`final_test.jsonl`** if they must reproduce the full **33,244-utterance** benchmark (not included in the repo due to size / distribution policy).

## Optional but helpful

4. **`baseline_summary_by_language.csv`** from a completed run (verification target after full eval).
5. Hugging Face: confirm they have been granted access to `ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large`.

## Their setup (short)

- Python **3.10**, CUDA GPU, `pip install -r requirements.txt`, AI4Bharat NeMo `nemo-v2` editable install.
- HF login, `stipa_key.json` on disk.
- Run **smoke test** first, then full eval with `final_test.jsonl`.

## Contacts

- **STIPA / Vaani manifests and GCS policy:** ask the dataset owner on your team (e.g. whoever maintains `final_test.jsonl`).
- **Baseline methodology questions:** see `stipa_asr_baseline_evaluation_report.md` and `README.md`.
