import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
REQUIRED = ["report_id", "report_text", "technique_id", "technique_name", "source_url", "label_evidence"]

def main():
    parser = argparse.ArgumentParser(description="Validate report data and create train/validation/test splits.")
    parser.add_argument("--input", default="data/sample_reports.csv")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.20, help="Fraction of remaining training pool used for validation.")
    args = parser.parse_args()

    path = (ROOT / args.input).resolve()
    df = pd.read_csv(path).fillna("")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    df = df[REQUIRED].copy()
    df["report_text"] = df["report_text"].astype(str).str.strip()
    df["technique_id"] = df["technique_id"].astype(str).str.strip()
    df = df[(df.report_text.str.len() > 20) & (df.technique_id != "")]
    df = df.drop_duplicates(subset=["report_text", "technique_id"]).reset_index(drop=True)

    if df.empty:
        raise SystemExit("No usable rows after validation.")
    counts = df["technique_id"].value_counts()
    # Stratification needs at least 2 examples per class for the first split.
    stratify = df["technique_id"] if counts.min() >= 2 and len(counts) > 1 else None
    trainval, test = train_test_split(
        df, test_size=args.test_size, random_state=42, stratify=stratify
    )
    train_counts = trainval["technique_id"].value_counts()
    stratify_trainval = trainval["technique_id"] if len(train_counts) > 1 and train_counts.min() >= 2 else None
    train, val = train_test_split(
        trainval, test_size=args.val_size, random_state=42, stratify=stratify_trainval
    )
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "validation.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    summary = {
        "input": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "rows_after_cleaning": int(len(df)),
        "unique_techniques": int(df.technique_id.nunique()),
        "technique_counts": {str(k): int(v) for k, v in counts.items()},
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "warning": "Synthetic demo data is only for pipeline smoke testing." if df.report_id.astype(str).str.startswith("SYN").all() else "Validate provenance and labels before research use."
    }
    import json
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
