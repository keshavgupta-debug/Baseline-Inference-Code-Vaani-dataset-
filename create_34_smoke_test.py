import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Build smoke_test.jsonl from a full test manifest")
    p.add_argument(
        "--input",
        type=str,
        default="final_test.jsonl",
        help="Full test JSONL manifest",
    )
    p.add_argument(
        "--output",
        type=str,
        default="smoke_test.jsonl",
        help="Output smoke manifest path",
    )
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    by_lang = defaultdict(list)
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            lang = data.get("language", "").strip().lower()
            by_lang[lang].append(data)

    short_clips = []
    long_clips = []

    print(f"Found {len(by_lang)} unique languages.\n")

    for lang in sorted(by_lang.keys()):
        clips = by_lang[lang]
        sorted_clips = sorted(clips, key=lambda x: float(x.get("duration", 0.0)))

        shortest = sorted_clips[0]
        longest = sorted_clips[-1]

        if len(sorted_clips) > 1 and shortest["audio_filepath"] == longest["audio_filepath"]:
            longest = sorted_clips[-2]

        short_clips.append(shortest)
        long_clips.append(longest)

    total_smoke = short_clips + long_clips

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in total_smoke:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Successfully generated {output_path} with {len(total_smoke)} clips (17 short, 17 long):\n")
    print(f"{'Language':15} | {'Shortest (s)':>12} | {'Longest (s)':>12} | {'Short File':>22} | {'Long File':>22}")
    print("-" * 92)
    for s, lng in zip(short_clips, long_clips):
        lang_name = s.get("language").lower()
        s_dur = f"{s.get('duration'):.2f}s"
        l_dur = f"{lng.get('duration'):.2f}s"
        s_fn = s.get("audio_filepath").split("/")[-1]
        l_fn = lng.get("audio_filepath").split("/")[-1]
        print(f"{lang_name:15} | {s_dur:>12} | {l_dur:>12} | {s_fn:>22} | {l_fn:>22}")


if __name__ == "__main__":
    main()
