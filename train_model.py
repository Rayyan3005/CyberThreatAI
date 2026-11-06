from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, set_seed
)

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "distilbert/distilbert-base-uncased"
MODEL_DIR = ROOT / "model" / "distilbert-attack-classifier"
OUT = ROOT / "outputs"
set_seed(42)

def read_split(name):
    p = OUT / f"{name}.csv"
    if not p.exists():
        raise SystemExit(f"{p} not found. Run: python prepare_data.py")
    df = pd.read_csv(p).fillna("")
    return df

def main():
    OUT.mkdir(exist_ok=True)
    train_df, val_df, test_df = read_split("train"), read_split("validation"), read_split("test")
    labels = sorted(train_df["technique_id"].astype(str).unique().tolist())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    # Single-label classification requires every evaluation label to exist in training.
    unseen = (set(val_df.technique_id.astype(str)) | set(test_df.technique_id.astype(str))) - set(labels)
    if unseen:
        raise SystemExit(f"Labels in validation/test missing from training: {sorted(unseen)}. Add more examples per class and rerun prepare_data.py.")

    names = {}
    for _, row in pd.concat([train_df, val_df, test_df]).iterrows():
        names[str(row.technique_id)] = str(row.technique_name)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    def convert(df):
        dataset = Dataset.from_pandas(pd.DataFrame({
            "text": df["report_text"].astype(str).tolist(),
            "labels": [label2id[str(x)] for x in df["technique_id"]]
        }))
        def tokenize(batch):
            return tokenizer(batch["text"], truncation=True, max_length=384)
        return dataset.map(tokenize, batched=True, remove_columns=["text"])

    train_ds, val_ds, test_ds = convert(train_df), convert(val_df), convert(test_df)

    def metrics(pred):
        logits, y_true = pred
        y_pred = np.argmax(logits, axis=-1)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(p),
            "macro_recall": float(r),
            "macro_f1": float(f1),
        }

    args = TrainingArguments(
        output_dir=str(ROOT / "outputs" / "training_checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        logging_steps=5,
        report_to="none",
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=metrics,
    )
    trainer.train()
    result = trainer.evaluate(test_ds)
    logits = trainer.predict(test_ds).predictions
    y_pred = np.argmax(logits, axis=-1)
    y_true = np.array([label2id[str(x)] for x in test_df["technique_id"]])
    target_names = [f"{label} — {names.get(label, 'Unknown technique')}" for label in labels]
    report = classification_report(
        y_true, y_pred, labels=list(range(len(labels))),
        target_names=target_names, zero_division=0
    )
    (OUT / "classification_report.txt").write_text(report, encoding="utf-8")
    metrics_out = {
        "model": MODEL_NAME,
        "task": "single-label technique classification",
        "dataset_rows": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "label2id": label2id,
        "technique_names": names,
        "test_metrics": {k: float(v) for k, v in result.items() if k.startswith("eval_")},
        "caveat": "If using sample_reports.csv, these are synthetic smoke-test results and must not be presented as real-world model performance."
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), max(5, len(labels))))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Test-set confusion matrix (inspect dataset provenance)")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    pred_df = test_df.copy()
    pred_df["predicted_technique_id"] = [id2label[int(i)] for i in y_pred]
    pred_df["predicted_technique_name"] = [names.get(id2label[int(i)], "Unknown") for i in y_pred]
    pred_df["correct"] = pred_df["technique_id"].astype(str) == pred_df["predicted_technique_id"]
    pred_df.to_csv(OUT / "test_predictions.csv", index=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    (MODEL_DIR / "technique_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    print("Training/evaluation finished.")
    print(json.dumps(metrics_out, indent=2))
    print(f"Saved model: {MODEL_DIR}")
    print(f"Saved metrics: {OUT / 'metrics.json'}")

if __name__ == "__main__":
    main()
