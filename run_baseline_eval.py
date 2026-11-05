"""
STIPA Baseline Model Evaluation Script
Model: ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large
Architecture: Conformer-Large Hybrid RNNT-CTC (AI4Bharat NeMo)
Evaluates Word Error Rate (WER) and Character Error Rate (CER) per language and overall.
"""

import os
import re
import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch
import pandas as pd
from tqdm import tqdm
from google.cloud import storage
import jiwer
import nemo.collections.asr as nemo_asr


def normalize_indic_text(text: str) -> str:
    """
    Standard text normalization for Indic ASR evaluation:
    - Removes punctuation (Devanagari danda ।, ॥, Latin punctuation, quotes, symbols)
    - Replaces zero-width joiners/non-joiners if isolated
    - Collapses multiple whitespace characters to a single space
    - Strips leading and trailing whitespace
    """
    if not text:
        return ""
    # Remove Devanagari danda and double danda
    text = text.replace("।", " ").replace("॥", " ")
    # Remove standard punctuation and symbols
    text = re.sub(r'[\.,\?!:;"\'\(\)\[\]\{\}\<\>\-\–\—\\\/_+=*&^%$#@~`|]', " ", text)
    # Remove zero-width characters
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_manifest(manifest_path: str):
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            lang = item.get("language", "unknown").strip().lower()
            records.append({
                "id": idx,
                "audio_filepath": item["audio_filepath"],
                "duration": float(item.get("duration", 0.0)),
                "text": item.get("text", ""),
                "language": lang,
                "variety": item.get("variety", lang).strip().lower(),
                "gender": item.get("gender", "unknown")
            })
    return records


def download_audio_files(records, cache_dir: str, gcs_client: storage.Client, max_workers: int = 16):
    os.makedirs(cache_dir, exist_ok=True)
    bucket_cache = {}
    
    def download_single(rec):
        gcs_uri = rec["audio_filepath"]
        # Expected format: gs://<bucket_name>/<blob_path>
        parts = gcs_uri.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        blob_path = parts[1]
        
        # Local file path
        local_filename = f"{rec['id']}_{Path(blob_path).name}"
        local_path = os.path.join(cache_dir, local_filename)
        rec["local_audio_path"] = local_path
        
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return local_path
        
        if bucket_name not in bucket_cache:
            bucket_cache[bucket_name] = gcs_client.bucket(bucket_name)
        bucket = bucket_cache[bucket_name]
        blob = bucket.blob(blob_path)
        blob.download_to_filename(local_path)
        return local_path

    print(f"[*] Downloading {len(records)} audio files to {cache_dir} using {max_workers} threads...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(download_single, records), total=len(records), desc="Audio Sync"))


def evaluate(manifest_path: str, key_path: str, output_dir: str, batch_size: int = 32, decoder: str = "rnnt", device_id: int = 0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    if key_path and os.path.exists(key_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(key_path)
        print(f"[+] Loaded GCS credentials from: {key_path}")
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cache_dir = out_path / "audio_cache"

    # Step 1: Parse Manifest
    print(f"[*] Loading manifest from {manifest_path}...")
    records = parse_manifest(manifest_path)
    total_hours = sum(r["duration"] for r in records) / 3600
    print(f"[+] Loaded {len(records):,} utterances ({total_hours:.2f} hours)")

    # Step 2: Download Audio Files
    gcs_client = storage.Client()
    download_audio_files(records, str(cache_dir), gcs_client)

    # Step 3: Load NeMo Baseline Model
    print(f"\n[*] Loading AI4Bharat Conformer Model (decoder={decoder})...")
    model_name = "ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large"
    
    # Check for downloaded .nemo files in cache or locally
    nemo_files = list(Path(os.path.expanduser("~/.cache/torch/NeMo")).glob("**/*.nemo")) or list(Path("./").glob("**/*.nemo"))
    if nemo_files:
        nemo_path = str(nemo_files[0])
        print(f"[+] Found cached .nemo checkpoint: {nemo_path}")
    else:
        from huggingface_hub import hf_hub_download
        print(f"[*] Downloading .nemo checkpoint from Hugging Face ({model_name})...")
        nemo_path = hf_hub_download(
            repo_id=model_name,
            filename="indicconformer_stt_hi_hybrid_rnnt_large.nemo"
        )
        print(f"[+] Downloaded checkpoint to: {nemo_path}")

    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(nemo_path)
    
    if hasattr(model, "cur_decoder"):
        model.cur_decoder = decoder
        print(f"[+] Active decoder set to: {model.cur_decoder}")
    
    model = model.cuda()
    model.eval()

    # Step 4: Batch Transcription
    print(f"\n[*] Starting batched transcription (batch_size={batch_size})...")
    audio_paths = [r["local_audio_path"] for r in records]
    
    start_time = time.time()
    with torch.no_grad():
        transcriptions = model.transcribe(
            audio=audio_paths,
            batch_size=batch_size,
            return_hypotheses=False,
            language_id="hi"
        )
        if isinstance(transcriptions, tuple):
            transcriptions = transcriptions[0]
    elapsed = time.time() - start_time
    print(f"[+] Inference complete in {elapsed/60:.2f} minutes (Speed: {total_hours / (elapsed/3600):.1f}x Real-Time)")

    # Step 5: Metric Computation (WER & CER)
    print("\n[*] Computing Word Error Rate (WER) and Character Error Rate (CER)...")
    results_by_lang = defaultdict(lambda: {"refs": [], "hyps": [], "dur": 0.0, "count": 0})
    all_refs = []
    all_hyps = []

    for rec, hyp_raw in zip(records, transcriptions):
        ref_norm = normalize_indic_text(rec["text"])
        hyp_norm = normalize_indic_text(hyp_raw)
        
        rec["pred_raw"] = hyp_raw
        rec["pred_norm"] = hyp_norm
        rec["ref_norm"] = ref_norm
        
        # Calculate individual sentence WER
        rec["sample_wer"] = jiwer.wer(ref_norm, hyp_norm) if ref_norm else (0.0 if not hyp_norm else 1.0)
        rec["sample_cer"] = jiwer.cer(ref_norm, hyp_norm) if ref_norm else (0.0 if not hyp_norm else 1.0)

        lang = rec["language"]
        results_by_lang[lang]["refs"].append(ref_norm)
        results_by_lang[lang]["hyps"].append(hyp_norm)
        results_by_lang[lang]["dur"] += rec["duration"]
        results_by_lang[lang]["count"] += 1

        all_refs.append(ref_norm)
        all_hyps.append(hyp_norm)

    # Step 6: Summary Table
    summary_rows = []
    for lang, data in sorted(results_by_lang.items(), key=lambda x: x[1]["dur"], reverse=True):
        lang_wer = jiwer.wer(data["refs"], data["hyps"]) * 100
        lang_cer = jiwer.cer(data["refs"], data["hyps"]) * 100
        summary_rows.append({
            "Language": lang,
            "Utterances": data["count"],
            "Hours": round(data["dur"] / 3600, 2),
            "WER (%)": round(lang_wer, 2),
            "CER (%)": round(lang_cer, 2)
        })

    overall_wer = jiwer.wer(all_refs, all_hyps) * 100
    overall_cer = jiwer.cer(all_refs, all_hyps) * 100

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 65)
    print("STIPA ASR BASELINE EVALUATION RESULTS")
    print(f"Model: {model_name} | Decoder: {decoder.upper()}")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print("-" * 65)
    print(f"OVERALL METRICS: WER = {overall_wer:.2f}% | CER = {overall_cer:.2f}% | Total Utterances = {len(records):,} | Hours = {total_hours:.2f}h")
    print("=" * 65)

    # Step 7: Export Artifacts
    df_details = pd.DataFrame(records)
    csv_details_path = out_path / "baseline_predictions_detailed.csv"
    summary_csv_path = out_path / "baseline_summary_by_language.csv"
    
    df_details.to_csv(csv_details_path, index=False, encoding="utf-8")
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8")
    
    print(f"\n[+] Detailed predictions saved to: {csv_details_path}")
    print(f"[+] Summary report saved to: {summary_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AI4Bharat Baseline ASR Model on STIPA Test Set")
    parser.add_argument("--manifest", type=str, required=True, help="Path to test jsonl manifest")
    parser.add_argument("--key", type=str, default="stipa_key.json", help="Path to GCS service account key")
    parser.add_argument("--output_dir", type=str, default="./stipa_baseline_eval", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--decoder", type=str, default="rnnt", choices=["rnnt", "ctc"], help="Decoder type")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index (CUDA_VISIBLE_DEVICES)")
    args = parser.parse_args()

    evaluate(
        manifest_path=args.manifest,
        key_path=args.key,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        decoder=args.decoder,
        device_id=args.gpu
    )
