"""
Streamlit app: evaluate 5 pre-trained classifiers on uploaded test data
(Breast Cancer Wisconsin Diagnostic dataset). Models are loaded from model/
and are never retrained here.

Run locally:  streamlit run app.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from train_models import TARGET, evaluate, model_filename

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_TEST_FILE = BASE_DIR / "test_data.csv"

MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Support Vector Machine",
]
CLASS_LABELS = ["Benign (0)", "Malignant (1)"]

st.set_page_config(page_title="Breast Cancer Classifier Comparison", layout="wide")


@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / model_filename(name)) for name in MODEL_NAMES}


@st.cache_resource
def load_feature_columns():
    return joblib.load(MODEL_DIR / "feature_columns.pkl")


def validate(df: pd.DataFrame, features: list) -> list:
    errors = []
    missing = [c for c in features if c not in df.columns]
    if missing:
        errors.append(f"Missing feature columns ({len(missing)}): {', '.join(missing[:10])}"
                      + (" ..." if len(missing) > 10 else ""))
    if TARGET not in df.columns:
        errors.append(f"Missing target column '{TARGET}'.")
    elif not set(df[TARGET].dropna().unique()) <= {0, 1}:
        errors.append(f"Target column '{TARGET}' must contain only 0 (benign) and 1 (malignant).")
    if df.isnull().any().any():
        errors.append("The file contains missing values; please clean it before uploading.")
    return errors


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Select model", MODEL_NAMES)

uploaded = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])
st.sidebar.caption(f"The CSV must contain all 30 feature columns and the target column '{TARGET}'.")

if SAMPLE_TEST_FILE.exists():
    st.sidebar.download_button(
        "Download sample test_data.csv",
        data=SAMPLE_TEST_FILE.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )
    use_sample = st.sidebar.checkbox("Use bundled sample test data instead")
else:
    use_sample = False

# ---------------- Main ----------------
st.title("Breast Cancer Diagnosis: Classifier Comparison")
st.write(
    "Evaluate five pre-trained classifiers (Logistic Regression, Decision Tree, KNN, "
    "Naive Bayes, SVM) on test data. Positive class = **malignant (1)**."
)

if uploaded is not None:
    data = pd.read_csv(uploaded)
    source = uploaded.name
elif use_sample:
    data = pd.read_csv(SAMPLE_TEST_FILE)
    source = "bundled test_data.csv"
else:
    st.info("Upload a test CSV from the sidebar (or tick 'Use bundled sample test data') to begin.")
    st.stop()

features = load_feature_columns()
errors = validate(data, features)
if errors:
    for e in errors:
        st.error(e)
    st.stop()

X = data[features]
y = data[TARGET].astype(int)
models = load_models()
model = models[model_name]

st.subheader(f"Data: {source}")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(data))
c2.metric("Malignant (1)", int((y == 1).sum()))
c3.metric("Benign (0)", int((y == 0).sum()))
with st.expander("Preview uploaded data"):
    st.dataframe(data.head(20))

# Metrics for the selected model
st.subheader(f"Evaluation metrics: {model_name}")
metrics = evaluate(model, X, y)
cols = st.columns(len(metrics))
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.4f}")

# Confusion matrix + classification report
y_pred = model.predict(X)
left, right = st.columns([1, 1.2])
with left:
    st.markdown("**Confusion matrix**")
    st.pyplot(plot_confusion_matrix(confusion_matrix(y, y_pred, labels=[0, 1])))
with right:
    st.markdown("**Classification report**")
    report = classification_report(y, y_pred, labels=[0, 1], target_names=CLASS_LABELS,
                                    output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(4))

# Comparison of all models on the same data
st.subheader("Comparison of all models on this data")
comparison = pd.DataFrame(
    [{"ML Model Name": name, **evaluate(m, X, y)} for name, m in models.items()]
).set_index("ML Model Name").round(4)
st.dataframe(
    comparison.style.highlight_max(axis=0, color="#c6efce").format("{:.4f}"),
)
best = comparison["F1"].idxmax()
st.success(f"Best model on this data by F1 score: **{best}** (F1 = {comparison.loc[best, 'F1']:.4f})")

# Predictions
with st.expander(f"Predictions from {model_name}"):
    out = data.copy()
    out["prediction"] = y_pred
    out["correct"] = out["prediction"] == y
    st.dataframe(out[[TARGET, "prediction", "correct"]])
    st.download_button("Download predictions CSV", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_filename(model_name)[:-4]}.csv",
                       mime="text/csv")
