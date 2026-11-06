
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL_DIR = ROOT / "model" / "distilbert-attack-classifier"

# Use the SAME splits that were used for the existing DistilBERT model
train_df = pd.read_csv(OUT / "train.csv").fillna("")
test_df = pd.read_csv(OUT / "test.csv").fillna("")

X_train = train_df["report_text"].astype(str)
y_train = train_df["technique_id"].astype(str)

X_test = test_df["report_text"].astype(str)
y_test = test_df["technique_id"].astype(str)

# 1. Train the baseline model on the existing training split
baseline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

baseline.fit(X_train, y_train)
# Save the trained baseline model
model_dir = ROOT / "model"
model_dir.mkdir(exist_ok=True)

joblib.dump(
    baseline,
    model_dir / "tfidf_logistic_regression.pkl"
)

print("Saved baseline model:", model_dir / "tfidf_logistic_regression.pkl")
baseline_pred = baseline.predict(X_test)


def calculate_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1
    }


baseline_metrics = calculate_metrics(y_test, baseline_pred)

# 2. Load the already-trained DistilBERT model
if not MODEL_DIR.exists():
    raise SystemExit(
        "Trained DistilBERT model not found. Run train_model.py first."
    )

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
bert_model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_DIR)
)
bert_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

bert_pred_ids = []

# Predict test reports in small batches
texts = X_test.tolist()

for start in range(0, len(texts), 8):
    batch = texts[start:start + 8]
    inputs = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=384,
        return_tensors="pt"
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = bert_model(**inputs).logits
        predictions = torch.argmax(logits, dim=-1)

    bert_pred_ids.extend(predictions.cpu().tolist())

# Convert model output IDs back to technique IDs
id2label = {
    int(key): str(value)
    for key, value in bert_model.config.id2label.items()
}
bert_pred = [id2label[int(i)] for i in bert_pred_ids]

bert_metrics = calculate_metrics(y_test, bert_pred)

# 3. Save a comparison table
comparison = pd.DataFrame([
    {"model": "TF-IDF + Logistic Regression", **baseline_metrics},
    {"model": "DistilBERT", **bert_metrics}
])

comparison_path = OUT / "model_comparison.csv"
comparison.to_csv(comparison_path, index=False)

print("\nMODEL COMPARISON — SAME TEST SET")
print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# 4. Save detailed classification reports
baseline_report = classification_report(
    y_test, baseline_pred, zero_division=0
)
bert_report = classification_report(
    y_test, bert_pred, zero_division=0
)

(OUT / "model_comparison_report.txt").write_text(
    "BASELINE: TF-IDF + LOGISTIC REGRESSION\n"
    + baseline_report
    + "\n\nDISTILBERT\n"
    + bert_report,
    encoding="utf-8"
)

# 5. Create a comparison graph
metric_columns = [
    "accuracy", "macro_precision", "macro_recall", "macro_f1"
]
ax = comparison.set_index("model")[metric_columns].T.plot(
    kind="bar", figsize=(10, 6), ylim=(0, 1.05)
)
ax.set_ylabel("Score")
ax.set_xlabel("Evaluation metric")
ax.set_title("Baseline vs DistilBERT — Same Test Set")
ax.legend(title="Model")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT / "model_comparison.png", dpi=160)
plt.close()

print("\nSaved files:")
print(comparison_path)
print(OUT / "model_comparison_report.txt")
print(OUT / "model_comparison.png")

print(
    "\nNote: Results on the small synthetic teaching dataset "
    "are demonstration results, not real-world performance."
)
