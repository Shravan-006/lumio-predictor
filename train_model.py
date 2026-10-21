"""
Model Training Script for Electricity Usage Predictor
Trains a scikit-learn DecisionTreeClassifier on household consumption features.

Key Design Decisions:
- Supervised Multi-Class Classification ('Low', 'Medium', 'High')
- Input Features: household_size, number_of_appliances, average_daily_usage_hours, ac_usage_hours_per_day
- Held-out 80/20 train-test split with stratification and fixed random_state (42)
- Constrained tree depth (max_depth=4) to prevent overfitting and ensure interpretability
- Saves fitted model to models/electricity_model.joblib
- Saves rich metadata to models/model_metadata.json
- Saves formatted report to reports/evaluation_results.txt
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------
# PATHS AND CONSTANTS
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "electricity_usage.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODEL_PATH = MODELS_DIR / "electricity_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
REPORT_PATH = REPORTS_DIR / "evaluation_results.txt"

FEATURE_COLUMNS = [
    "household_size",
    "number_of_appliances",
    "average_daily_usage_hours",
    "ac_usage_hours_per_day",
]
TARGET_COLUMN = "usage_category"
CLASS_LABELS = ["Low", "Medium", "High"]
RANDOM_STATE = 42
TEST_SIZE = 0.20


def train_and_evaluate():
    print("=" * 60)
    print("Training Decision Tree Classifier: Electricity Usage Predictor")
    print("=" * 60)

    # 1. Ensure directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Check dataset presence
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Please run 'python generate_dataset.py' first."
        )

    # 3. Load and validate dataset
    df = pd.read_csv(DATA_PATH)
    print(f"[OK] Loaded dataset: {len(df)} records from {DATA_PATH.name}")

    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    if df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().any().any():
        print("[WARNING] Null values detected; dropping null rows.")
        df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Verify target values
    unique_targets = set(y.unique())
    expected_targets = set(CLASS_LABELS)
    if not unique_targets.issubset(expected_targets):
        raise ValueError(f"Unexpected target labels: {unique_targets - expected_targets}")

    # 4. Train-Test Split (Stratified 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"[OK] Split completed: {len(X_train)} training samples, {len(X_test)} testing samples")

    # 5. Model Initialization and Training
    # max_depth=4 balances expressiveness and generalization, preventing overfitting
    # min_samples_split=8 and min_samples_leaf=4 provide robust regularization
    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        min_samples_split=8,
        min_samples_leaf=4,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    print("[OK] DecisionTreeClassifier fitted successfully.")

    # 6. Evaluation on held-out test subset
    y_pred = clf.predict(X_test)
    y_train_pred = clf.predict(X_train)

    train_accuracy = float(accuracy_score(y_train, y_train_pred))
    test_accuracy = float(accuracy_score(y_test, y_pred))

    precision_macro = float(precision_score(y_test, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_test, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0))

    precision_weighted = float(precision_score(y_test, y_pred, labels=CLASS_LABELS, average="weighted", zero_division=0))
    recall_weighted = float(recall_score(y_test, y_pred, labels=CLASS_LABELS, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, labels=CLASS_LABELS, average="weighted", zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)
    report_dict = classification_report(y_test, y_pred, labels=CLASS_LABELS, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, labels=CLASS_LABELS, zero_division=0)

    feature_importances = {
        feat: round(float(imp), 4)
        for feat, imp in zip(FEATURE_COLUMNS, clf.feature_importances_)
    }

    # 7. Print Results to Console
    print("\n--- Model Evaluation Results ---")
    print(f"Training Accuracy : {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy     : {test_accuracy * 100:.2f}%")
    print(f"Precision (Macro) : {precision_macro * 100:.2f}%")
    print(f"Recall (Macro)    : {recall_macro * 100:.2f}%")
    print(f"F1-Score (Macro)  : {f1_macro * 100:.2f}%")
    print(f"F1-Score (Weight) : {f1_weighted * 100:.2f}%")
    print("\nConfusion Matrix (Labels: ['Low', 'Medium', 'High']):")
    print(cm)
    print("\nClassification Report:\n")
    print(report_text)
    print("Feature Importances:")
    for feat, imp in feature_importances.items():
        print(f"  - {feat:28s}: {imp * 100:.2f}%")

    # 8. Save Fitted Model
    joblib.dump(clf, MODEL_PATH)
    print(f"\n[OK] Model serialized to: {MODEL_PATH}")

    # 9. Save Metadata JSON
    metadata = {
        "model_type": "DecisionTreeClassifier",
        "algorithm": "Decision Tree (CART via Gini Impurity)",
        "feature_names": FEATURE_COLUMNS,
        "target_name": TARGET_COLUMN,
        "class_labels": CLASS_LABELS,
        "hyperparameters": {
            "criterion": "gini",
            "max_depth": 4,
            "min_samples_split": 8,
            "min_samples_leaf": 4,
            "random_state": RANDOM_STATE,
        },
        "dataset_statistics": {
            "total_records": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "class_distribution": {
                label: int((df[TARGET_COLUMN] == label).sum()) for label in CLASS_LABELS
            },
        },
        "metrics": {
            "train_accuracy": round(train_accuracy, 4),
            "test_accuracy": round(test_accuracy, 4),
            "precision_macro": round(precision_macro, 4),
            "recall_macro": round(recall_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "precision_weighted": round(precision_weighted, 4),
            "recall_weighted": round(recall_weighted, 4),
            "f1_weighted": round(f1_weighted, 4),
        },
        "confusion_matrix": {
            "labels": CLASS_LABELS,
            "matrix": cm.tolist(),
        },
        "classification_report": report_dict,
        "feature_importances": feature_importances,
        "thresholds_kwh": {
            "low_threshold": 10.0,
            "high_threshold": 22.0,
        },
        "synthetic_data_notice": (
            "Model trained on simulated educational data. For demonstration purposes only."
        ),
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Metadata saved to: {METADATA_PATH}")

    # 10. Save Text Evaluation Report
    report_content = f"""============================================================
ELECTRICITY USAGE CATEGORY PREDICTOR - MODEL EVALUATION REPORT
============================================================
Generated by train_model.py
Algorithm: DecisionTreeClassifier (scikit-learn)
Hyperparameters: criterion=gini, max_depth=4, min_samples_split=8, min_samples_leaf=4
Random State: {RANDOM_STATE} (Reproducible)

DATASET INFORMATION:
- Total Dataset Size: {len(df)} records
- Training Set Size : {len(X_train)} records (80%)
- Testing Set Size  : {len(X_test)} records (20%)
- Input Features    : {', '.join(FEATURE_COLUMNS)}
- Target Label      : {TARGET_COLUMN} {CLASS_LABELS}

PERFORMANCE METRICS (Held-Out Test Set):
------------------------------------------------------------
Training Accuracy       : {train_accuracy * 100:.2f}%
Test Accuracy           : {test_accuracy * 100:.2f}%
Macro Precision         : {precision_macro * 100:.2f}%
Macro Recall            : {recall_macro * 100:.2f}%
Macro F1-Score          : {f1_macro * 100:.2f}%
Weighted F1-Score       : {f1_weighted * 100:.2f}%

CONFUSION MATRIX:
Labels: {CLASS_LABELS}
{cm}

DETAILED CLASSIFICATION REPORT:
{report_text}

FEATURE IMPORTANCE RANKINGS:
"""
    for feat, imp in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True):
        report_content += f"- {feat:28s}: {imp * 100:.2f}%\n"

    report_content += """
============================================================
NOTE ON SIMULATED DATA:
This evaluation reflects performance on a synthetic educational dataset.
Decision boundary thresholds correspond to physical rules embedded in data generation,
augmented with Gaussian noise. While performance demonstrates accurate rule induction,
these metrics cannot be interpreted as validation on real-world utility meters.
============================================================
"""

    with open(REPORT_PATH, "w") as f:
        f.write(report_content)
    print(f"[OK] Text report saved to: {REPORT_PATH}")
    print("=" * 60)
    print("Model Training and Evaluation Pipeline Finished Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    train_and_evaluate()
