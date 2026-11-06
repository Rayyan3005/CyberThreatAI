# CyberThreatAI — SOC Threat-Intelligence Tagging Assistant

**Project title:** An Intelligent System to Auto-Tag Threat-Intelligence Reports by Attack Technique

This starter project provides a reproducible pipeline to prepare labelled report text, fine-tune a transformer classifier, evaluate it, and use it through a Streamlit SOC-assistant interface.

## Important limitations
- This is a research/educational decision-support prototype, not a production SOC detection system.
- The included `data/sample_reports.csv` is **synthetic demonstration data**. It is not a public threat-intelligence corpus and must not be described as real-world evidence.
- Model metrics from a tiny synthetic dataset are only a pipeline smoke test, not meaningful evidence of generalisation.
- ATT&CK labels in any real corpus must be validated against source reports and the MITRE ATT&CK version used.
- The starter implements **single-label classification**. Real reports often describe multiple techniques; a validated multi-label extension may be more appropriate.
- The inference UI can show a predicted technique and score, but scores are not guaranteed calibrated probabilities and predictions require analyst verification.

## Folder layout
- `data/sample_reports.csv` — synthetic smoke-test dataset
- `data/real_reports_template.csv` — blank template for properly sourced reports
- `prepare_data.py` — validation and split creation
- `train_model.py` — fine-tunes DistilBERT and writes evaluation outputs
- `app.py` — Streamlit SOC assistant
- `fetch_attack.py` — downloads the official MITRE ATT&CK Enterprise STIX bundle for reference
- `outputs/` — generated reports, metrics and plots
- `model/` — saved fine-tuned model

## Windows setup (CMD)
Open CMD and run:
```cmd
D:
cd \CyberThreatAI
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
If `.\.venv\Scripts\activate.bat` does not exist or Python paths are inconsistent, recreate the environment using a known Python installation and then install requirements. Check `where python` and `python --version`.

## 1. Prepare data
The real-data CSV needs these columns:
- `report_id`: stable unique ID
- `report_text`: report text or an appropriately licensed excerpt
- `technique_id`: MITRE ATT&CK technique ID, e.g. `T1566.001`
- `technique_name`: matching technique name
- `source_url`: source report URL
- `label_evidence`: brief note describing why the label is supported

Start with the synthetic demo only to test the pipeline:
```cmd
python prepare_data.py --input data\sample_reports.csv
```

For real research, populate `data\real_reports_template.csv` with sourced and manually validated labels. Do not infer labels from keywords alone. Keep source URL, retrieval date, ATT&CK version, and labeling rationale in your research log.

## 2. Download MITRE ATT&CK reference data
Internet required:
```cmd
python fetch_attack.py
```
The script saves the official Enterprise ATT&CK STIX bundle to `data\enterprise-attack.json`. Review MITRE's terms and current documentation before redistributing data.

## 3. Train and evaluate
Training downloads a pretrained DistilBERT model, so internet access is required on first run:
```cmd
python train_model.py
```
The tiny synthetic dataset is intended only for a smoke test; it may fail a stratified split or produce meaningless scores. Use a sufficiently large, independently reviewed dataset with enough examples per class for research-quality results.

Generated files:
- `outputs/metrics.json`
- `outputs/classification_report.txt`
- `outputs/confusion_matrix.png`
- `outputs/test_predictions.csv`
- model files under `model/distilbert-attack-classifier/`

Do not present smoke-test metrics as research findings.

## 4. Run the SOC assistant
```cmd
streamlit run app.py
```
Choose the saved model directory, paste a report, and inspect the predicted tag and score. The UI displays a warning that predictions need analyst verification.

## Evaluation protocol recommendations
- Split by report/source or campaign when possible to reduce near-duplicate leakage.
- Keep a held-out test set untouched until final evaluation.
- Report macro-F1 and per-class precision/recall/F1, plus class counts.
- Compare with a simple TF-IDF + Logistic Regression baseline before claiming the transformer is better.
- Record dataset retrieval date, ATT&CK version, inclusion/exclusion criteria, deduplication rules, and annotation procedure.
