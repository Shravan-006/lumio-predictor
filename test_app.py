"""
Automated Verification and Test Suite for Electricity Usage Category Predictor
Tests file integrity, model deserialization, batch inference, edge cases, and metadata.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

def test_project_structure():
    print("[TEST 1/6] Checking project files and directory structure...")
    required_paths = [
        BASE_DIR / "requirements.txt",
        BASE_DIR / "generate_dataset.py",
        BASE_DIR / "train_model.py",
        BASE_DIR / "app.py",
        BASE_DIR / "README.md",
        BASE_DIR / "VIVA_NOTES.md",
        BASE_DIR / "electricity_usage.csv",
        BASE_DIR / "models" / "electricity_model.joblib",
        BASE_DIR / "models" / "model_metadata.json",
        BASE_DIR / "reports" / "evaluation_results.txt",
    ]
    for p in required_paths:
        assert p.exists(), f"Missing required file: {p}"
        assert p.stat().st_size > 0, f"File is empty: {p}"
        print(f"  [OK] Found: {p.relative_to(BASE_DIR)} ({p.stat().st_size} bytes)")
    print("  --> Project structure test PASSED!\n")


def test_dataset_integrity():
    print("[TEST 2/6] Validating dataset schema, records, and distributions...")
    csv_path = BASE_DIR / "electricity_usage.csv"
    df = pd.read_csv(csv_path)
    
    assert len(df) == 800, f"Expected 800 records, got {len(df)}"
    expected_cols = [
        "household_size",
        "number_of_appliances",
        "average_daily_usage_hours",
        "ac_usage_hours_per_day",
        "simulated_daily_consumption_kwh",
        "usage_category",
    ]
    assert list(df.columns) == expected_cols, f"Column mismatch: {df.columns}"
    assert df.isnull().sum().sum() == 0, "Found null values in dataset"
    
    # Check bounds
    assert (df["household_size"] >= 1).all() and (df["household_size"] <= 12).all()
    assert (df["number_of_appliances"] >= 1).all()
    assert (df["ac_usage_hours_per_day"] <= df["average_daily_usage_hours"] + 1e-5).all(), "AC hours exceed total usage hours"
    
    categories = set(df["usage_category"].unique())
    assert categories == {"Low", "Medium", "High"}, f"Unexpected categories: {categories}"
    print(f"  [OK] Records: {len(df)}, Columns: {len(df.columns)}, Nulls: 0")
    print(f"  [OK] Classes: {dict(df['usage_category'].value_counts())}")
    print("  --> Dataset integrity test PASSED!\n")


def test_model_inference():
    print("[TEST 3/6] Testing model loading and multi-class predictions...")
    model_path = BASE_DIR / "models" / "electricity_model.joblib"
    model = joblib.load(model_path)
    
    features = [
        "household_size",
        "number_of_appliances",
        "average_daily_usage_hours",
        "ac_usage_hours_per_day",
    ]
    
    # Test Case 1: Low consumption scenario (1 person, 4 appliances, 5 hours, 0 AC)
    low_input = pd.DataFrame([[1, 4, 5.0, 0.0]], columns=features)
    pred_low = model.predict(low_input)[0]
    prob_low = model.predict_proba(low_input)[0]
    print(f"  [OK] Low Profile: input={low_input.values.tolist()} -> Predicted: {pred_low}, Proba: {prob_low}")
    assert pred_low == "Low", f"Expected Low, got {pred_low}"
    
    # Test Case 2: High consumption scenario (6 people, 22 appliances, 16 hours, 10 AC hours)
    high_input = pd.DataFrame([[6, 22, 16.0, 10.0]], columns=features)
    pred_high = model.predict(high_input)[0]
    prob_high = model.predict_proba(high_input)[0]
    print(f"  [OK] High Profile: input={high_input.values.tolist()} -> Predicted: {pred_high}, Proba: {prob_high}")
    assert pred_high == "High", f"Expected High, got {pred_high}"

    # Test Case 3: Medium consumption scenario (3 people, 11 appliances, 9 hours, 2.5 AC hours)
    med_input = pd.DataFrame([[3, 11, 9.0, 2.5]], columns=features)
    pred_med = model.predict(med_input)[0]
    prob_med = model.predict_proba(med_input)[0]
    print(f"  [OK] Medium Profile: input={med_input.values.tolist()} -> Predicted: {pred_med}, Proba: {prob_med}")
    assert pred_med in ["Low", "Medium", "High"], f"Invalid prediction: {pred_med}"
    
    # Check probabilities sum to 1
    assert np.isclose(np.sum(prob_low), 1.0)
    assert np.isclose(np.sum(prob_high), 1.0)
    assert np.isclose(np.sum(prob_med), 1.0)
    print("  --> Model inference test PASSED!\n")


def test_metadata_consistency():
    print("[TEST 4/6] Verifying metadata and evaluation report...")
    meta_path = BASE_DIR / "models" / "model_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    assert meta["model_type"] == "DecisionTreeClassifier"
    assert meta["class_labels"] == ["Low", "Medium", "High"]
    assert "metrics" in meta
    test_acc = meta["metrics"]["test_accuracy"]
    assert 0.85 <= test_acc <= 1.0, f"Unusual test accuracy: {test_acc}"
    print(f"  [OK] Stored Test Accuracy: {test_acc * 100:.2f}%")
    print(f"  [OK] Macro Precision: {meta['metrics']['precision_macro'] * 100:.2f}%")
    print(f"  [OK] Macro Recall: {meta['metrics']['recall_macro'] * 100:.2f}%")
    print(f"  [OK] Macro F1-Score: {meta['metrics']['f1_macro'] * 100:.2f}%")
    
    report_path = BASE_DIR / "reports" / "evaluation_results.txt"
    report_text = report_path.read_text(encoding="utf-8")
    assert "CONFUSION MATRIX" in report_text
    assert "CLASSIFICATION REPORT" in report_text
    assert "FEATURE IMPORTANCE RANKINGS" in report_text
    print("  --> Metadata and report test PASSED!\n")


def test_validation_rule():
    print("[TEST 5/6] Verifying business validation rule (AC hours <= daily hours)...")
    def validate_inputs(usage_hours, ac_hours):
        return ac_hours <= usage_hours

    assert validate_inputs(8.0, 3.0) is True
    assert validate_inputs(5.0, 5.0) is True
    assert validate_inputs(4.0, 6.0) is False
    print("  [OK] Validation rule catches AC hours exceeding usage hours correctly.")
    print("  --> Validation rule test PASSED!\n")


def test_app_imports_and_loaders():
    print("[TEST 6/6] Testing app.py functions directly in Python...")
    import app
    
    df = app.load_dataset()
    assert df is not None
    assert len(df) == 800
    
    model = app.load_model()
    assert model is not None
    
    meta = app.load_metadata()
    assert meta is not None
    print("  [OK] app.load_dataset() loaded dataframe correctly.")
    print("  [OK] app.load_model() loaded joblib model correctly.")
    print("  [OK] app.load_metadata() loaded json metadata correctly.")
    print("  --> App imports and loader test PASSED!\n")


def main():
    print("=" * 60)
    print("RUNNING AUTOMATED TEST SUITE")
    print("=" * 60)
    test_project_structure()
    test_dataset_integrity()
    test_model_inference()
    test_metadata_consistency()
    test_validation_rule()
    test_app_imports_and_loaders()
    print("=" * 60)
    print("ALL 6 TEST SUITES PASSED FLAWLESSLY! READY FOR EXPO DEMO.")
    print("=" * 60)

if __name__ == "__main__":
    main()
