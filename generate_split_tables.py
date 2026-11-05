import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Print markdown distribution tables for STIPA splits")
    p.add_argument("--data-dir", type=str, default="./data", help="Directory containing split JSONL files")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    splits = [
        ("Train Set (All)", data_dir / "final_train_all.jsonl", "Train Set %"),
        ("Train Set (Script Head)", data_dir / "final_train_script.jsonl", "Script Set %"),
        ("Validation Set", data_dir / "final_val.jsonl", "Val Set %"),
        ("Test Set", data_dir / "final_test.jsonl", "Test Set %"),
    ]

    for title, path, pct_col in splits:
        if not path.exists():
            print(f"\n[skip] Missing: {path}")
            continue

        lang_data = defaultdict(lambda: {"count": 0, "dur": 0.0, "male": 0, "female": 0, "other": 0})
        total_dur = 0.0
        total_count = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                l = d.get("language", "unknown").strip().lower()
                dur = float(d.get("duration", 0.0))
                g = str(d.get("gender", "")).strip().capitalize()

                lang_data[l]["count"] += 1
                lang_data[l]["dur"] += dur
                total_dur += dur
                total_count += 1

                if g == "Male":
                    lang_data[l]["male"] += 1
                elif g == "Female":
                    lang_data[l]["female"] += 1
                else:
                    lang_data[l]["other"] += 1

        print(f"\n### {title} (Total: {total_count:,} Utterances | {total_dur/3600:.2f} Hours)")
        print(f"| Language | Utterances | Hours | {pct_col} | Male / Female |")
        print(f"| :--- | :---: | :---: | :---: | :---: |")

        for l, s in sorted(lang_data.items(), key=lambda x: x[1]["dur"], reverse=True):
            hrs = s["dur"] / 3600.0
            pct = (hrs / (total_dur / 3600.0)) * 100 if total_dur > 0 else 0
            m = s["male"]
            f = s["female"]
            o = s["other"]
            gender_str = f"{m:,}/{f:,}" + (f" (and {o} other)" if o else "")
            print(f"| **{l.capitalize()}** | {s['count']:,} | {hrs:.2f}h | {pct:.2f}% | {gender_str} |")


if __name__ == "__main__":
    main()
