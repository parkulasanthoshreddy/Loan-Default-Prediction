
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline


# ----- PATHS -----
PROJECT_DIR = Path(r"C:\Users\Santhosh\OneDrive\Desktop\projects\Loan Default Prediction")
DATA_FILE   = PROJECT_DIR / r"dataset\loan.csv"
OUT_DIR     = PROJECT_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----- READ DATA (robust encoding) -----
try:
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(DATA_FILE, encoding="latin1")

# ----- TARGET DETECTION -----
cols_lower = {c.lower(): c for c in df.columns}

# Explicitly handle "Credit Default"
if "credit default" in cols_lower:
    target_col = cols_lower["credit default"]

elif "default" in cols_lower:
    target_col = cols_lower["default"]

elif "loan_status" in cols_lower:
    status_col = cols_lower["loan_status"]
    bad_keywords = ["charge", "default", "late", "bad", "write", "overdue", "delinq"]
    df["__target__"] = df[status_col].astype(str).str.lower().apply(
        lambda x: 1 if any(k in x for k in bad_keywords) else 0
    )
    target_col = "__target__"
else:
    raise SystemExit("Could not find target. Add a 'default' (0/1) column or a 'loan_status' text column.")

# X / y
y = df[target_col].astype(int)
X = df.drop(columns=[target_col])

# Drop obvious id-like columns
drop_id_like = [c for c in X.columns if any(k in c.lower() for k in ["id", "member_id", "application_id", "uuid"])]
if drop_id_like:
    X = X.drop(columns=drop_id_like)

# Identify types
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# Preprocessors
num_pre = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler(with_mean=False))  # sparse compatibility
])

cat_pre = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

pre = ColumnTransformer(
    transformers=[
        ("num", num_pre, num_cols),
        ("cat", cat_pre, cat_cols),
    ],
    remainder="drop"
)

# (Optional) 

models = {
    "logreg": ImbPipeline([
        ("prep", pre),
        # ("smote", smote),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=None))
    ]),
    "random_forest": ImbPipeline([
        ("prep", pre),
        # ("smote", smote),
        ("clf", RandomForestClassifier(
            n_estimators=400, max_depth=None, random_state=42, class_weight="balanced_subsample", n_jobs=-1
        ))
    ])
}

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

results = {}
roc_curves = {}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        y_proba = None
        auc = None

    report = classification_report(y_test, y_pred, output_dict=True)
    report["auc"] = auc
    results[name] = report

    # Confusion matrix
    plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"confusion_matrix_{name}.png", dpi=160)
    plt.close()

    # ROC curve
    if y_proba is not None:
        plt.figure()
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title(f"ROC Curve — {name} (AUC={auc:.3f})")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"roc_{name}.png", dpi=160)
        plt.close()

# Save metrics
with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

summary_rows = []
for name, rpt in results.items():
    row = {
        "model": name,
        "accuracy": rpt.get("accuracy"),
        "precision_0": rpt.get("0", {}).get("precision"),
        "recall_0": rpt.get("0", {}).get("recall"),
        "precision_1": rpt.get("1", {}).get("precision"),
        "recall_1": rpt.get("1", {}).get("recall"),
        "f1_1": rpt.get("1", {}).get("f1-score"),
        "auc": rpt.get("auc")
    }
    summary_rows.append(row)
pd.DataFrame(summary_rows).to_csv(OUT_DIR / "metrics_summary.csv", index=False)

# Feature importance (RF) & coefficients (LogReg)
def export_model_details():
    # get fitted transformers to map feature names
    prep = models["random_forest"].named_steps["prep"]
    ohe = prep.named_transformers_["cat"].named_steps["ohe"]
    # numeric feature names (after imputer/scale remain same)
    num_names = num_cols
    cat_names = list(ohe.get_feature_names_out(cat_cols)) if len(cat_cols) else []
    all_names = list(num_names) + cat_names

    # RandomForest importances
    rf = models["random_forest"].named_steps["clf"]
    if hasattr(rf, "feature_importances_"):
        fi = pd.DataFrame({
            "feature": all_names,
            "importance": rf.feature_importances_
        }).sort_values("importance", ascending=False)
        fi.to_csv(OUT_DIR / "feature_importances_rf.csv", index=False)

    # LogisticRegression coefficients
    lr = models["logreg"].named_steps["clf"]
    if hasattr(lr, "coef_"):
        coefs = pd.DataFrame({
            "feature": all_names,
            "coef": lr.coef_[0]
        }).sort_values("coef", ascending=False)
        coefs.to_csv(OUT_DIR / "coefficients_logreg.csv", index=False)

try:
    export_model_details()
except Exception as e:
    print("Note: could not export feature names/coefs:", e)

print(" Done. See outputs folder:", OUT_DIR.resolve())
