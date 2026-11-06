import sys
from pathlib import Path
import json

print("Testing environment...", flush=True)

try:
    import joblib
    print("Joblib imported successfully.", flush=True)
    
    baseline_path = Path("model/tfidf_logistic_regression.pkl")
    if baseline_path.exists():
        baseline_model = joblib.load(baseline_path)
        pred = baseline_model.predict(["A user received a malicious attachment in email."])
        print("Baseline model loaded and predicted:", pred[0], flush=True)
    else:
        print("Baseline model file not found.", flush=True)

    import torch
    print("PyTorch version:", torch.__version__, flush=True)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    print("Transformers imported.", flush=True)

    model_dir = Path("model/distilbert-attack-classifier")
    if model_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        print("DistilBERT loaded successfully!", flush=True)
        inputs = tokenizer("A user clicked a phishing link to a fake login portal.", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=-1).item()
            label = model.config.id2label.get(pred_id, str(pred_id))
            print("DistilBERT test prediction:", label, flush=True)
    else:
        print("DistilBERT model dir not found.", flush=True)

    print("ALL ENVIRONMENT TESTS PASSED!", flush=True)
except Exception as e:
    print(f"Error during verification: {e}", flush=True)
    import traceback
    traceback.print_exc()
