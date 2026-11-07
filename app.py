
from pathlib import Path
import html
import json
import re

import joblib
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =========================================================
# 1. PATHS AND PAGE CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "model" / "distilbert-attack-classifier"
BASELINE_MODEL = ROOT / "model" / "tfidf_logistic_regression.pkl"

st.set_page_config(
    page_title="CyberThreatAI | Threat Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. DARK CYBERSECURITY THEME
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #0B1220;
        color: #E5EDF7;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 18, 32, 0.96);
    }

    [data-testid="stSidebar"] {
        background: #101A2B;
        border-right: 1px solid #26364B;
    }

    [data-testid="stSidebar"] * {
        color: #E5EDF7;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(120deg, #112238, #102D3C);
        border: 1px solid #245366;
        border-radius: 20px;
        padding: 27px 30px;
        margin-bottom: 22px;
    }

    .hero-title {
        color: #F1F7FF;
        font-size: clamp(25px, 3vw, 36px);
        font-weight: 800;
        margin: 0 0 8px 0;
        line-height: 1.3;
    }

    .hero-subtitle, .muted-text {
        color: #A8BDCF;
        font-size: 14px;
        line-height: 1.7;
    }

    .eyebrow {
        color: #5EEAD4;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }

    .section-heading {
        color: #67E8F9;
        font-size: 21px;
        font-weight: 750;
        margin-top: 18px;
        margin-bottom: 12px;
    }

    .prediction-card, .review-card {
        background: #111D30;
        border: 1px solid #2B4058;
        border-radius: 14px;
        padding: 17px;
        margin: 10px 0;
        overflow-wrap: anywhere;
    }

    .prediction-rank {
        color: #67E8F9;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    .prediction-name {
        color: #F0F7FF;
        font-size: 17px;
        font-weight: 750;
        line-height: 1.5;
    }

    .prediction-id {
        color: #9DB4C9;
        font-size: 13px;
        margin-top: 5px;
    }

    .score-label {
        color: #9DB4C9;
        font-size: 12px;
        margin-top: 13px;
    }

    .score-value {
        color: #67E8F9;
        font-size: 26px;
        font-weight: 800;
        margin: 2px 0 8px;
    }

    .score-track {
        height: 7px;
        background: #26364B;
        border-radius: 20px;
        overflow: hidden;
    }

    .score-fill {
        height: 7px;
        background: linear-gradient(90deg, #0891B2, #34D399);
        border-radius: 20px;
    }

    .review-title {
        color: #F1F7FF;
        font-size: 16px;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .review-status {
        font-size: 12px;
        font-weight: 800;
        letter-spacing: .4px;
        margin-bottom: 7px;
    }

    .status-pass { color: #86EFAC; }
    .status-warn { color: #FCD34D; }
    .status-info { color: #67E8F9; }

    div[data-testid="stButton"] > button {
        min-height: 46px;
        border-radius: 11px;
        font-weight: 700;
        border: 1px solid #34516A;
        background: #15243A;
        color: #E8F4FF;
    }

    div[data-testid="stButton"] > button:hover {
        border-color: #22D3EE;
        color: #67E8F9;
    }

    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(100deg, #0891B2, #0D9488);
        color: white;
        border: 1px solid #22D3EE;
    }

    .stTextArea textarea {
        background: #101B2C !important;
        color: #EDF6FF !important;
        border: 1px solid #344A62 !important;
        border-radius: 12px !important;
        line-height: 1.65 !important;
    }

    .stTextArea label, .stRadio label, .stSlider label {
        color: #D8E5F2 !important;
    }

    [data-testid="stMetric"] {
        background: #111D30;
        border: 1px solid #2B4058;
        border-radius: 12px;
        padding: 12px;
    }

    [data-testid="stMetricValue"] {
        color: #67E8F9;
        overflow-wrap: anywhere;
    }

    [data-testid="stExpander"] {
        border: 1px solid #2B4058;
        border-radius: 12px;
        background: #0F1A2B;
    }

    hr {
        border-color: #26364B;
    }

    @media (max-width: 700px) {
        .block-container {
            padding: 1rem .8rem 2rem;
        }

        .hero {
            padding: 20px 17px;
        }

        .prediction-card, .review-card {
            padding: 13px;
        }

        .prediction-name {
            font-size: 15px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. TECHNIQUE MAPPING AND EVIDENCE RULES
# =========================================================

TECHNIQUES = {
    "T1078": {
        "name": "Valid Accounts",
        "patterns": [
            r"\bvalid account(?:s)?\b",
            r"\bstolen credentials?\b",
            r"\bstolen username and password\b",
            r"\bcompromised account(?:s)?\b",
            r"\bstolen login credentials?\b",
            r"\blogged in using (?:stolen|compromised) credentials\b",
            r"\bunauthori[sz]ed login\b",
            r"\baccount takeover\b",
        ],
    },
    "T1566.001": {
        "name": "Spearphishing Attachment",
        "patterns": [
            r"\bspearphishing attachment\b",
            r"\bmalicious attachment\b",
            r"\bphishing email.{0,100}\battachment\b",
            r"\bemail.{0,80}\battached (?:document|file|spreadsheet)\b",
            r"\battached (?:document|file|spreadsheet).{0,100}\bmalicious\b",
            r"\bopens? the attachment\b",
            r"\battachment.{0,80}\bexecutes?\b",
        ],
    },
    "T1566.002": {
        "name": "Spearphishing Link",
        "patterns": [
            r"\bspearphishing link\b",
            r"\bphishing link\b",
            r"\bdeceptive hyperlink\b",
            r"\bmalicious link\b",
            r"\bfake (?:company |corporate )?(?:sign[- ]in|login) page\b",
            r"\bcredential harvesting page\b",
            r"\bemail.{0,100}\blink.{0,100}\bcredentials?\b",
            r"\bclicks? the link\b",
        ],
    },
}


# =========================================================
# 4. AUTOMATIC PRELIMINARY REVIEW
# =========================================================

def review_threat(report_text, prediction):
    text = report_text.lower()
    technique_id = str(prediction["id"])
    technique_name = str(prediction["name"])

    evidence = {}

    for tech_id, details in TECHNIQUES.items():
        evidence[tech_id] = any(
            re.search(pattern, text, flags=re.IGNORECASE)
            for pattern in details["patterns"]
        )

    id_valid = technique_id in TECHNIQUES

    expected_name = (
        TECHNIQUES[technique_id]["name"] if id_valid else None
    )

    name_valid = (
        id_valid
        and technique_name.strip().casefold()
        == expected_name.casefold()
    )

    top_evidence_found = bool(evidence.get(technique_id, False))

    indicated_techniques = [
        tech_id for tech_id, found in evidence.items() if found
    ]

    return {
        "id": technique_id,
        "name": technique_name,
        "id_valid": id_valid,
        "name_valid": name_valid,
        "top_evidence_found": top_evidence_found,
        "indicated_techniques": indicated_techniques,
        "multiple_indicators": len(indicated_techniques) > 1,
    }


def show_review_card(title, status, message, css_class="status-info"):
    st.markdown(
        f"""
        <div class="review-card">
            <div class="review-title">{html.escape(title)}</div>
            <div class="review-status {css_class}">
                {html.escape(status)}
            </div>
            <div class="muted-text">{html.escape(message)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_automatic_review(report_text, prediction):
    review = review_threat(report_text, prediction)

    st.markdown(
        f'<div class="section-heading">Review: '
        f'{html.escape(review["id"])} — '
        f'{html.escape(review["name"])}</div>',
        unsafe_allow_html=True,
    )

    if review["id_valid"]:
        show_review_card(
            "1. Technique ID validation",
            "SUPPORTED BY PROJECT MAPPING",
            f'{review["id"]} exists in this prototype mapping. '
            "This is not independent verification of the official ATT&CK release.",
            "status-pass",
        )
    else:
        show_review_card(
            "1. Technique ID validation",
            "REVIEW REQUIRED",
            f'{review["id"]} is not included in this prototype mapping.',
            "status-warn",
        )

    if review["name_valid"]:
        show_review_card(
            "2. Technique name validation",
            "ID AND NAME MATCH",
            f'{review["id"]} maps to {review["name"]} in the project mapping.',
            "status-pass",
        )
    else:
        show_review_card(
            "2. Technique name validation",
            "REVIEW REQUIRED",
            "The predicted ID and name do not match the supported mapping.",
            "status-warn",
        )

    if review["top_evidence_found"]:
        show_review_card(
            "3. Report evidence indicator",
            "RELATED PHRASES FOUND",
            "Configured phrases associated with this technique were found. "
            "A phrase match does not confirm an attack.",
            "status-pass",
        )
    else:
        show_review_card(
            "3. Report evidence indicator",
            "NO CONFIGURED PHRASE MATCH",
            "No configured phrase was found for this prediction. "
            "This does not prove the prediction is wrong.",
            "status-warn",
        )

    if review["multiple_indicators"]:
        ids = ", ".join(review["indicated_techniques"])
        show_review_card(
            "4. Possible multiple techniques",
            "MULTIPLE INDICATORS FOUND",
            f"Configured phrases matched: {ids}. Review the report to "
            "determine whether multiple techniques are actually described.",
            "status-warn",
        )
    elif review["indicated_techniques"]:
        ids = ", ".join(review["indicated_techniques"])
        show_review_card(
            "4. Possible multiple techniques",
            "ONE TECHNIQUE INDICATOR FOUND",
            f"The configured rules matched {ids}. Other techniques may "
            "be present even if these rules do not detect them.",
            "status-info",
        )
    else:
        show_review_card(
            "4. Possible multiple techniques",
            "NO CONFIGURED INDICATORS FOUND",
            "No supported technique indicators were matched. This does not "
            "mean that the report has no attack techniques.",
            "status-warn",
        )

    show_review_card(
        "5. Final analyst verification",
        "REQUIRED",
        "An analyst must review the evidence and context before assigning "
        "a final label. These preliminary checks do not confirm an incident.",
        "status-warn",
    )

    st.caption(
        "Heuristic checks cover T1078, T1566.001 and T1566.002 only. "
        "They may produce false positives or false negatives."
    )


# =========================================================
# 5. MODEL LOADING
# =========================================================

MODEL_RELEASE_URL = (
    "https://github.com/Rayyan3005/CyberThreatAI/releases/download/v1.0.0/model.safetensors"
)


def download_file_with_progress(url: str, dest_path: Path):
    """Download model weights with a visual progress bar if running in cloud deployment."""
    import urllib.request

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.info(
        "⬇️ Downloading fine-tuned DistilBERT weights from GitHub Releases (~255MB)... "
        "This runs once on cloud deployment and is cached."
    )

    def report_hook(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(int(downloaded * 100 / total_size), 100)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            progress_bar.progress(
                percent,
                text=f"Downloading DistilBERT weights: {percent}% ({mb_down:.1f} MB / {mb_total:.1f} MB)",
            )

    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_dest = dest_path.with_suffix(".tmp")
        urllib.request.urlretrieve(url, temp_dest, reporthook=report_hook)
        if temp_dest.exists() and temp_dest.stat().st_size > 1000000:
            temp_dest.replace(dest_path)
            progress_bar.empty()
            status_text.empty()
            return True
        else:
            if temp_dest.exists():
                temp_dest.unlink()
            progress_bar.empty()
            status_text.empty()
            return False
    except Exception as exc:
        progress_bar.empty()
        status_text.empty()
        st.warning(f"Note: Could not automatically fetch DistilBERT weights: {exc}")
        return False


@st.cache_resource(show_spinner="Loading DistilBERT...")
def load_transformer(model_path):
    path = Path(model_path)
    weights_path = path / "model.safetensors"
    config_path = path / "config.json"

    # In cloud environments (e.g. Streamlit Cloud), auto-download model.safetensors if missing
    if not weights_path.exists() or weights_path.stat().st_size < 1000000:
        download_file_with_progress(MODEL_RELEASE_URL, weights_path)

    if not path.exists() or not config_path.exists() or not weights_path.exists():
        return None, None, {}

    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForSequenceClassification.from_pretrained(str(path))
    model.eval()

    names_file = path / "technique_names.json"
    names = (
        json.loads(names_file.read_text(encoding="utf-8"))
        if names_file.exists()
        else {}
    )

    return tokenizer, model, names


@st.cache_resource(show_spinner="Loading TF-IDF model...")
def load_baseline(model_path):
    path = Path(model_path)

    if not path.exists():
        return None

    return joblib.load(path)


# =========================================================
# 6. PREDICTION FUNCTIONS
# =========================================================

def predict_transformer(text, tokenizer, model, names, top_k):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=384,
    )

    with torch.no_grad():
        scores = torch.softmax(
            model(**inputs).logits[0], dim=-1
        ).cpu().numpy()

    predictions = []

    for idx, score in enumerate(scores):
        label = str(model.config.id2label.get(int(idx), str(idx)))
        name = names.get(label, label)

        if label in TECHNIQUES:
            name = TECHNIQUES[label]["name"]

        predictions.append({
            "id": label,
            "name": name,
            "score": float(score),
        })

    return sorted(
        predictions,
        key=lambda item: item["score"],
        reverse=True,
    )[:top_k]


def predict_baseline(text, model, top_k):
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_

    predictions = []

    for idx, score in enumerate(probabilities):
        label = str(classes[idx])
        name = TECHNIQUES.get(label, {}).get("name", label)

        predictions.append({
            "id": label,
            "name": name,
            "score": float(score),
        })

    return sorted(
        predictions,
        key=lambda item: item["score"],
        reverse=True,
    )[:top_k]


# =========================================================
# 7. DISPLAY PREDICTIONS
# =========================================================

def show_predictions(title, predictions):
    st.markdown(
        f'<div class="section-heading">{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )

    for rank, item in enumerate(predictions, start=1):
        score_percent = max(
            0.0, min(100.0, item["score"] * 100)
        )

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-rank">PREDICTION {rank}</div>
                <div class="prediction-name">
                    {html.escape(str(item["name"]))}
                </div>
                <div class="prediction-id">
                    ATT&CK Technique ID: {html.escape(str(item["id"]))}
                </div>
                <div class="score-label">Model score</div>
                <div class="score-value">{score_percent:.1f}%</div>
                <div class="score-track">
                    <div class="score-fill"
                         style="width: {score_percent:.1f}%;">
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# 8. REVIEW EXPANDERS: CLOSED BY DEFAULT
# =========================================================

def show_review_expanders(title, report_text, predictions):
    st.markdown(
        f'<div class="section-heading">{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )

    # Only the predictions returned by top_k are displayed.
    for prediction in predictions:
        label = (
            f'{prediction["id"]} — {prediction["name"]}'
        )

        # IMPORTANT: expanded=False keeps every review closed initially.
        with st.expander(label, expanded=False):
            show_automatic_review(report_text, prediction)


# =========================================================
# 9. HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">SOC • THREAT INTELLIGENCE • AI</div>
        <div class="hero-title">🛡️ CyberThreatAI</div>
        <p class="hero-subtitle">
            Intelligent system to automatically tag threat-intelligence
            reports by MITRE ATT&CK technique.
        </p>
        <p class="hero-subtitle">● Research Prototype</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Decision-support only. Predictions can be wrong and must be verified "
    "by a security analyst. Use only public or non-sensitive report text "
    "unless authorised."
)


# =========================================================
# 10. LOAD MODELS
# =========================================================

try:
    tokenizer, transformer_model, technique_names = load_transformer(
        DEFAULT_MODEL
    )
except Exception as exc:
    tokenizer, transformer_model, technique_names = None, None, {}

try:
    baseline_model = load_baseline(BASELINE_MODEL)
except Exception as exc:
    baseline_model = None


# =========================================================
# 11. SIDEBAR CONFIGURATION
# =========================================================

with st.sidebar:
    st.markdown("## 🛡️ CyberThreatAI")
    st.caption("Threat analysis workspace")
    st.markdown("---")

    st.markdown("### Model Configuration")

    model_options = []
    default_idx = 0
    if transformer_model is not None:
        model_options.append("DistilBERT")
    if baseline_model is not None:
        model_options.append("TF-IDF + Logistic Regression")

    if not model_options:
        model_options = ["DistilBERT", "TF-IDF + Logistic Regression"]
    elif transformer_model is None:
        default_idx = 0

    selected_model = st.radio(
        "Choose a model",
        model_options,
        index=default_idx,
    )

    top_k = st.slider(
        "Number of tags to show",
        min_value=1,
        max_value=5,
        value=3,
    )

    st.markdown("---")
    st.markdown("### Model Information")
    distil_status = "🟢 Ready" if transformer_model is not None else "⚠️ Unavailable"
    tfidf_status = "🟢 Ready" if baseline_model is not None else "⚠️ Unavailable"
    st.caption(f"**DistilBERT**: Transformer-based classifier ({distil_status})")
    st.caption(f"**TF-IDF**: Text features + Logistic Regression ({tfidf_status})")
    st.info("Scores are model outputs, not calibrated probabilities.")


# Status notices on main area if a model is missing
if transformer_model is None and baseline_model is not None:
    st.info(
        "ℹ️ **DistilBERT weights are downloading or offline.** "
        "The app is currently operating with the **TF-IDF + Logistic Regression** baseline model. "
        "You can analyze reports right away."
    )
elif transformer_model is None and baseline_model is None:
    st.error("No models are available. Check the model folder and saved files.")


# =========================================================
# 12. REPORT INPUT AND ANALYSIS
# =========================================================

if transformer_model is not None or baseline_model is not None:
    st.markdown(
        '<div class="section-heading">Threat Report Analysis</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p class="muted-text">
            Paste a threat report to generate predictions and automatic
            preliminary review indicators.
        </p>
        """,
        unsafe_allow_html=True,
    )

    report_text = st.text_area(
        "Threat-intelligence report",
        height=230,
        placeholder="Paste a public, non-sensitive report excerpt here...",
        help="Enter at least 30 characters.",
    )

    col1, col2 = st.columns(2)

    analyze_clicked = col1.button(
        "🔍 Analyze with selected model",
        type="primary",
        use_container_width=True,
    )

    compare_clicked = col2.button(
        "⚖️ Compare Both Models",
        use_container_width=True,
    )

    if analyze_clicked or compare_clicked:

        if len(report_text.strip()) < 30:
            st.error("Please enter at least 30 characters of report text.")

        else:
            # ---------------------------------------------
            # ANALYZE SELECTED MODEL
            # ---------------------------------------------

            if analyze_clicked:
                if selected_model == "DistilBERT":

                    if tokenizer is None or transformer_model is None:
                        st.error("DistilBERT is unavailable.")
                    else:
                        with st.spinner("Analysing with DistilBERT..."):
                            results = predict_transformer(
                                report_text,
                                tokenizer,
                                transformer_model,
                                technique_names,
                                top_k,
                            )

                        show_predictions("DistilBERT Predictions", results)

                        show_review_expanders(
                            "Automatic Preliminary Analyst Review",
                            report_text,
                            results,
                        )

                else:
                    if baseline_model is None:
                        st.error("TF-IDF model is unavailable.")
                    else:
                        with st.spinner("Analysing with TF-IDF..."):
                            results = predict_baseline(
                                report_text, baseline_model, top_k
                            )

                        show_predictions(
                            "TF-IDF + Logistic Regression Predictions",
                            results,
                        )

                        show_review_expanders(
                            "Automatic Preliminary Analyst Review",
                            report_text,
                            results,
                        )

            # ---------------------------------------------
            # COMPARE BOTH MODELS
            # ---------------------------------------------

            if compare_clicked:
                if tokenizer is None or transformer_model is None:
                    st.error("Cannot compare: DistilBERT is unavailable.")

                elif baseline_model is None:
                    st.error("Cannot compare: TF-IDF model is unavailable.")

                else:
                    with st.spinner("Comparing both models..."):
                        distil_results = predict_transformer(
                            report_text,
                            tokenizer,
                            transformer_model,
                            technique_names,
                            top_k,
                        )

                        tfidf_results = predict_baseline(
                            report_text, baseline_model, top_k
                        )

                    st.markdown("---")
                    st.markdown(
                        '<div class="section-heading">'
                        'Side-by-Side Model Comparison'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    left, right = st.columns(2, gap="large")

                    with left:
                        show_predictions("DistilBERT", distil_results)

                    with right:
                        show_predictions(
                            "TF-IDF + Logistic Regression",
                            tfidf_results,
                        )

                    distil_top = distil_results[0]["id"]
                    tfidf_top = tfidf_results[0]["id"]

                    if distil_top == tfidf_top:
                        st.success(
                            "Both models predict the same top technique: "
                            f"{distil_top}. Analyst verification is still required."
                        )
                    else:
                        st.info(
                            "The models disagree on the top prediction. "
                            "Review the report before assigning a final label."
                        )

                    st.caption(
                        "Results are demonstrations from a small synthetic "
                        "dataset. Scores are not calibrated probabilities "
                        "and do not establish real-world performance."
                    )

                    st.markdown("---")

                    review_left, review_right = st.columns(
                        2, gap="large"
                    )

                    with review_left:
                        show_review_expanders(
                            "DistilBERT Review",
                            report_text,
                            distil_results,
                        )

                    with review_right:
                        show_review_expanders(
                            "TF-IDF Review",
                            report_text,
                            tfidf_results,
                        )


# =========================================================
# 13. FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; padding:10px 0;">
        <div style="color:#67E8F9; font-weight:800; font-size:15px;">
            🛡️ CyberThreatAI
        </div>
        <div style="color:#8FA6BC; font-size:12px; margin-top:5px;">
            AI-assisted threat intelligence • Research prototype
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
