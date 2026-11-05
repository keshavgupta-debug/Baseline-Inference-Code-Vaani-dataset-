import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path


def pearson_corr(x, y):
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den_x = sum((xi - mean_x) ** 2 for xi in x)
    den_y = sum((yi - mean_y) ** 2 for yi in y)
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / math.sqrt(den_x * den_y)


def default_manifest_paths(data_dir: Path):
    return {
        "train": data_dir / "final_train_all.jsonl",
        "script": data_dir / "final_train_script.jsonl",
        "val": data_dir / "final_val.jsonl",
        "test": data_dir / "final_test.jsonl",
    }


def parse_args():
    p = argparse.ArgumentParser(description="Cross-split language distribution vs baseline WER/CER")
    p.add_argument("--data-dir", type=str, default="./data", help="Directory containing split JSONL files")
    p.add_argument(
        "--baseline-summary",
        type=str,
        default="./stipa_full_eval/baseline_summary_by_language.csv",
        help="Baseline summary CSV from run_baseline_eval.py",
    )
    p.add_argument(
        "--output",
        type=str,
        default="./cross_split_distribution_and_correlations.csv",
        help="Output comparative CSV path",
    )
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    paths = default_manifest_paths(data_dir)
    baseline_summary_path = args.baseline_summary
    output_csv = args.output

    baseline_metrics = {}
    if not os.path.exists(baseline_summary_path):
        raise FileNotFoundError(
            f"Baseline summary not found: {baseline_summary_path}. "
            "Run run_baseline_eval.py first or pass --baseline-summary."
        )
    with open(baseline_summary_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = row["Language"].strip().lower()
            baseline_metrics[lang] = {
                "wer": float(row["WER (%)"]),
                "cer": float(row["CER (%)"]),
            }

    splits = {
        "train": str(paths["train"]),
        "script": str(paths["script"]),
        "val": str(paths["val"]),
        "test": str(paths["test"]),
    }

    stats = {s: defaultdict(lambda: {"count": 0, "duration": 0.0}) for s in splits}
    totals = {s: {"count": 0, "duration": 0.0} for s in splits}

    for sname, path in splits.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing manifest: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                lang = d.get("language", "unknown").strip().lower()
                dur = float(d.get("duration", 0.0))
                stats[sname][lang]["count"] += 1
                stats[sname][lang]["duration"] += dur
                totals[sname]["count"] += 1
                totals[sname]["duration"] += dur

    all_langs = sorted(
        list(stats["test"].keys()),
        key=lambda l: stats["test"][l]["duration"],
        reverse=True,
    )

    matrix = []
    for l in all_langs:
        tr_h = stats["train"][l]["duration"] / 3600.0
        tr_pct = (stats["train"][l]["duration"] / totals["train"]["duration"] * 100) if totals["train"]["duration"] > 0 else 0

        sc_h = stats["script"][l]["duration"] / 3600.0
        sc_pct = (stats["script"][l]["duration"] / totals["script"]["duration"] * 100) if totals["script"]["duration"] > 0 else 0

        val_h = stats["val"][l]["duration"] / 3600.0
        val_pct = (stats["val"][l]["duration"] / totals["val"]["duration"] * 100) if totals["val"]["duration"] > 0 else 0

        test_h = stats["test"][l]["duration"] / 3600.0
        test_pct = (stats["test"][l]["duration"] / totals["test"]["duration"] * 100) if totals["test"]["duration"] > 0 else 0

        wer = baseline_metrics.get(l, {}).get("wer", None)
        cer = baseline_metrics.get(l, {}).get("cer", None)
        in_train = tr_h > 0

        matrix.append(
            {
                "language": l,
                "in_train": in_train,
                "train_utts": stats["train"][l]["count"],
                "train_hours": tr_h,
                "train_pct": tr_pct,
                "script_utts": stats["script"][l]["count"],
                "script_hours": sc_h,
                "script_pct": sc_pct,
                "val_utts": stats["val"][l]["count"],
                "val_hours": val_h,
                "val_pct": val_pct,
                "test_utts": stats["test"][l]["count"],
                "test_hours": test_h,
                "test_pct": test_pct,
                "wer": wer,
                "cer": cer,
            }
        )

    test_hours_list = [m["test_hours"] for m in matrix if m["wer"] is not None]
    wer_list = [m["wer"] for m in matrix if m["wer"] is not None]
    cer_list = [m["cer"] for m in matrix if m["wer"] is not None]

    corr_test_wer = pearson_corr(test_hours_list, wer_list)
    corr_test_cer = pearson_corr(test_hours_list, cer_list)

    train_hours_list = [m["train_hours"] for m in matrix if m["wer"] is not None]
    corr_train_wer = pearson_corr(train_hours_list, wer_list)
    corr_train_cer = pearson_corr(train_hours_list, cer_list)

    seen_wers = [m["wer"] for m in matrix if m["in_train"] and m["wer"] is not None]
    seen_cers = [m["cer"] for m in matrix if m["in_train"] and m["cer"] is not None]
    tail_wers = [m["wer"] for m in matrix if not m["in_train"] and m["wer"] is not None]
    tail_cers = [m["cer"] for m in matrix if not m["in_train"] and m["cer"] is not None]

    mean_seen_wer = sum(seen_wers) / len(seen_wers) if seen_wers else 0
    mean_seen_cer = sum(seen_cers) / len(seen_cers) if seen_cers else 0
    mean_tail_wer = sum(tail_wers) / len(tail_wers) if tail_wers else 0
    mean_tail_cer = sum(tail_cers) / len(tail_cers) if tail_cers else 0

    print("=== CORRELATION ANALYSIS ===")
    print(f"Correlation (Test Hours vs Baseline WER): {corr_test_wer:.4f}")
    print(f"Correlation (Test Hours vs Baseline CER): {corr_test_cer:.4f}")
    print(f"Correlation (Train Hours vs Baseline WER): {corr_train_wer:.4f}")
    print(f"Correlation (Train Hours vs Baseline CER): {corr_train_cer:.4f}")
    print(f"\nSeen in Train (7 languages): Mean WER = {mean_seen_wer:.2f}% | Mean CER = {mean_seen_cer:.2f}%")
    print(f"Tail / Zero-Shot (10 languages): Mean WER = {mean_tail_wer:.2f}% | Mean CER = {mean_tail_cer:.2f}%")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Language",
                "Split Group",
                "Train Utts",
                "Train Hours",
                "Train % of Split",
                "Script Utts",
                "Script Hours",
                "Script % of Split",
                "Val Utts",
                "Val Hours",
                "Val % of Split",
                "Test Utts",
                "Test Hours",
                "Test % of Split",
                "Baseline WER (%)",
                "Baseline CER (%)",
            ]
        )
        for m in matrix:
            group = "Train+Test (Core)" if m["in_train"] else "Test/Val Only (Tail)"
            writer.writerow(
                [
                    m["language"],
                    group,
                    m["train_utts"],
                    f"{m['train_hours']:.2f}",
                    f"{m['train_pct']:.2f}%",
                    m["script_utts"],
                    f"{m['script_hours']:.2f}",
                    f"{m['script_pct']:.2f}%",
                    m["val_utts"],
                    f"{m['val_hours']:.2f}",
                    f"{m['val_pct']:.2f}%",
                    m["test_utts"],
                    f"{m['test_hours']:.2f}",
                    f"{m['test_pct']:.2f}%",
                    m["wer"],
                    m["cer"],
                ]
            )
    print(f"\nSaved cross-split table to: {output_csv}")


if __name__ == "__main__":
    main()
