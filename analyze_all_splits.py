import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path


def default_manifest_paths(data_dir: Path):
    return {
        "train_all": data_dir / "final_train_all.jsonl",
        "train_script": data_dir / "final_train_script.jsonl",
        "val": data_dir / "final_val.jsonl",
        "test": data_dir / "final_test.jsonl",
    }


def parse_args():
    p = argparse.ArgumentParser(description="Profile STIPA manifests across train/val/test splits")
    p.add_argument("--data-dir", type=str, default="./data", help="Directory containing split JSONL files")
    p.add_argument(
        "--test-manifest",
        type=str,
        default=None,
        help="Override test manifest path (default: <data-dir>/final_test.jsonl)",
    )
    p.add_argument(
        "--val-manifest",
        type=str,
        default=None,
        help="Override validation manifest path",
    )
    p.add_argument(
        "--train-all-manifest",
        type=str,
        default=None,
        help="Override train-all manifest path",
    )
    p.add_argument(
        "--train-script-manifest",
        type=str,
        default=None,
        help="Override train-script manifest path",
    )
    p.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Optional final_metadata.csv path (default: <data-dir>/final_metadata.csv)",
    )
    p.add_argument(
        "--baseline-summary",
        type=str,
        default="./stipa_full_eval/baseline_summary_by_language.csv",
        help="Baseline summary CSV from run_baseline_eval.py",
    )
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    paths = default_manifest_paths(data_dir)

    test_path = args.test_manifest or str(paths["test"])
    val_path = args.val_manifest or str(paths["val"])
    train_all_path = args.train_all_manifest or str(paths["train_all"])
    train_script_path = args.train_script_manifest or str(paths["train_script"])
    metadata_path = args.metadata or str(data_dir / "final_metadata.csv")
    baseline_summary_path = args.baseline_summary

    baseline_metrics = {}
    if os.path.exists(baseline_summary_path):
        with open(baseline_summary_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lang = row["Language"].strip().lower()
                baseline_metrics[lang] = {
                    "wer": float(row["WER (%)"]),
                    "cer": float(row["CER (%)"]),
                    "test_utts": int(row["Utterances"]),
                    "test_hours": float(row["Hours"]),
                }

    splits = {
        "Train (All)": train_all_path,
        "Train (Script Head)": train_script_path,
        "Validation": val_path,
        "Test": test_path,
    }

    split_data = {}

    for split_name, path in splits.items():
        lang_stats = defaultdict(lambda: {"count": 0, "duration": 0.0, "male": 0, "female": 0})
        total_utts = 0
        total_dur = 0.0

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    lang = d.get("language", "unknown").strip().lower()
                    dur = float(d.get("duration", 0.0))
                    gender = str(d.get("gender", "")).strip().lower()

                    lang_stats[lang]["count"] += 1
                    lang_stats[lang]["duration"] += dur
                    if gender == "male":
                        lang_stats[lang]["male"] += 1
                    elif gender == "female":
                        lang_stats[lang]["female"] += 1

                    total_utts += 1
                    total_dur += dur
        else:
            print(f"[warn] Missing manifest: {path}")

        split_data[split_name] = {
            "total_utts": total_utts,
            "total_hours": total_dur / 3600.0,
            "langs": lang_stats,
        }

    print("=== SPLIT TOTALS ===")
    for sname, sinfo in split_data.items():
        print(
            f"{sname:22}: {sinfo['total_utts']:,} utts | "
            f"{sinfo['total_hours']:.2f} hours | {len(sinfo['langs'])} languages"
        )

    all_langs = sorted(list(set(lang for s in split_data.values() for lang in s["langs"].keys())))
    print(f"\nTotal Unique Languages Across All Splits: {len(all_langs)}")

    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            print(f"\nMetadata CSV Columns: {header}")
            row_count = sum(1 for _ in reader)
            print(f"Metadata CSV Total Rows: {row_count:,}")
    elif metadata_path:
        print(f"\n[warn] Metadata not found: {metadata_path}")


if __name__ == "__main__":
    main()
