
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "sample_reports.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset using the actual CSV column names
df = pd.read_csv(DATA_FILE)
df = df.dropna(subset=["report_text", "technique_id"])

X = df["report_text"].astype(str)
y = df["technique_id"].astype(str)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# TF-IDF + Logistic Regression baseline
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
report = classification_report(
    y_test,
    predictions,
    zero_division=0
)

print("\nBASELINE MODEL: TF-IDF + LOGISTIC REGRESSION")
print(f"Dataset rows: {len(df)}")
print(f"Training rows: {len(X_train)}")
print(f"Test rows: {len(X_test)}")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)

report_path = OUTPUT_DIR / "baseline_classification_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("Baseline Model: TF-IDF + Logistic Regression\n")
    file.write(f"Dataset rows: {len(df)}\n")
    file.write(f"Training rows: {len(X_train)}\n")
    file.write(f"Test rows: {len(X_test)}\n")
    file.write(f"Accuracy: {accuracy:.4f}\n\n")
    file.write(report)

print(f"\nReport saved to: {report_path}")
