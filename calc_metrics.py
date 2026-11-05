import argparse
import jiwer
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Recalculate WER/CER from baseline prediction CSV")
    p.add_argument(
        "--predictions",
        type=str,
        default="./stipa_full_eval/baseline_predictions_detailed.csv",
        help="Path to baseline_predictions_detailed.csv",
    )
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.predictions)
    print(f"Total evaluated utterances: {len(df):,}")
    total_hours = df["duration"].sum() / 3600
    print(f"Total duration: {total_hours:.2f} hours")

    refs = df["ref_norm"].fillna("").astype(str).tolist()
    hyps = df["pred_norm"].fillna("").astype(str).tolist()

    overall_wer = jiwer.wer(refs, hyps) * 100
    overall_cer = jiwer.cer(refs, hyps) * 100
    print(f"\nOVERALL CORPUS METRICS:")
    print(f"  WER: {overall_wer:.2f}%")
    print(f"  CER: {overall_cer:.2f}%")

    hindi_df = df[df["language"] == "hindi"]
    hindi_refs = hindi_df["ref_norm"].fillna("").astype(str).tolist()
    hindi_hyps = hindi_df["pred_norm"].fillna("").astype(str).tolist()
    hindi_wer = jiwer.wer(hindi_refs, hindi_hyps) * 100
    hindi_cer = jiwer.cer(hindi_refs, hindi_hyps) * 100
    print(f"\nSTANDARD HINDI SUBSET ({len(hindi_df):,} utts, {hindi_df['duration'].sum()/3600:.2f}h):")
    print(f"  WER: {hindi_wer:.2f}%")
    print(f"  CER: {hindi_cer:.2f}%")

    regional_df = df[df["language"] != "hindi"]
    regional_refs = regional_df["ref_norm"].fillna("").astype(str).tolist()
    regional_hyps = regional_df["pred_norm"].fillna("").astype(str).tolist()
    regional_wer = jiwer.wer(regional_refs, regional_hyps) * 100
    regional_cer = jiwer.cer(regional_refs, regional_hyps) * 100
    print(
        f"\nREGIONAL VARIETIES / SISTER LANGUAGES ({len(regional_df):,} utts, "
        f"{regional_df['duration'].sum()/3600:.2f}h):"
    )
    print(f"  WER: {regional_wer:.2f}%")
    print(f"  CER: {regional_cer:.2f}%")


if __name__ == "__main__":
    main()
