"""
Electricity Usage Dataset Generator
Simulates educational household electricity consumption data for ML classification.

NOTE ON SIMULATED DATA:
This dataset is synthetically generated for educational and demonstration purposes
in a B.Tech CSE project. It does NOT represent genuine smart-meter or physical sensor measurements.
Predictive results derived from this synthetic dataset should be viewed as an educational
demonstration of machine learning concepts, not as an industrial energy audit.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# CONFIGURATION & REPRODUCIBILITY
# ---------------------------------------------------------
RANDOM_SEED = 42
NUM_SAMPLES = 800

# Consumption classification thresholds (kWh per day)
# Low: < 10.0 kWh/day (Approx. < 300 kWh/month)
# Medium: 10.0 to 22.0 kWh/day (Approx. 300 to 660 kWh/month)
# High: > 22.0 kWh/day (Approx. > 660 kWh/month)
LOW_THRESHOLD_KWH = 10.0
HIGH_THRESHOLD_KWH = 22.0

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "electricity_usage.csv"


def generate_electricity_dataset(n_samples: int = NUM_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generates synthetic household records with realistic correlations.
    
    Features:
    - household_size: 1 to 8 persons
    - number_of_appliances: 3 to 25 appliances
    - average_daily_usage_hours: 4.0 to 18.0 hours
    - ac_usage_hours_per_day: 0.0 to 14.0 hours (constrained <= average_daily_usage_hours)
    
    Target generation:
    - simulated_daily_consumption_kwh = baseline + appliances_load + ac_load + realistic_noise
    - usage_category: 'Low', 'Medium', 'High'
    """
    np.random.seed(seed)
    
    # We simulate 3 archetypes to ensure balanced representation of realistic households:
    # 1. Compact / Low-demand households (~30%)
    # 2. Average / Moderate-demand households (~40%)
    # 3. High-demand / Large households (~30%)
    n_compact = int(n_samples * 0.32)
    n_average = int(n_samples * 0.38)
    n_high = n_samples - n_compact - n_average
    
    records = []
    
    # Archetype 1: Compact (1-3 members, 3-8 appliances, lower hours, low/zero AC)
    for _ in range(n_compact):
        size = np.random.randint(1, 4)
        appliances = np.random.randint(3, 9)
        usage_hours = np.round(np.random.uniform(4.0, 9.0), 1)
        # AC usage: mostly 0 to 2 hours
        ac_hours = np.round(np.random.choice([0.0, 0.0, 0.5, 1.0, 2.0]), 1)
        ac_hours = min(ac_hours, usage_hours)
        records.append((size, appliances, usage_hours, ac_hours))
        
    # Archetype 2: Moderate (2-5 members, 8-16 appliances, moderate hours, moderate AC)
    for _ in range(n_average):
        size = np.random.randint(2, 6)
        appliances = np.random.randint(8, 17)
        usage_hours = np.round(np.random.uniform(7.0, 14.0), 1)
        ac_hours = np.round(np.random.uniform(1.0, 6.0), 1)
        ac_hours = min(ac_hours, usage_hours)
        records.append((size, appliances, usage_hours, ac_hours))
        
    # Archetype 3: High Demand (4-8 members, 14-25 appliances, higher hours, high AC)
    for _ in range(n_high):
        size = np.random.randint(4, 9)
        appliances = np.random.randint(14, 26)
        usage_hours = np.round(np.random.uniform(10.0, 18.0), 1)
        ac_hours = np.round(np.random.uniform(5.0, 14.0), 1)
        ac_hours = min(ac_hours, usage_hours)
        records.append((size, appliances, usage_hours, ac_hours))
        
    df = pd.DataFrame(records, columns=[
        "household_size",
        "number_of_appliances",
        "average_daily_usage_hours",
        "ac_usage_hours_per_day"
    ])
    
    # Shuffle records to avoid ordering bias
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    # ---------------------------------------------------------
    # SIMULATED CONSUMPTION FORMULA (PHYSICALLY MOTIVATED)
    # ---------------------------------------------------------
    # Base load per person (lighting, refrigeration share, phone charging): ~1.25 kWh
    # General appliance load: ~0.35 kWh per appliance normalized around 8 operating hours
    # Air Conditioner load (1.5 ton split AC consumes ~1.4 - 1.6 kWh per active hour): ~1.45 kWh/hr
    # Realistic noise: Gaussian variance N(0, 1.2^2) to simulate efficiency differences
    noise = np.random.normal(0.0, 1.2, size=len(df))
    
    calculated_kwh = (
        1.25 * df["household_size"]
        + 0.38 * df["number_of_appliances"] * (df["average_daily_usage_hours"] / 8.0)
        + 1.45 * df["ac_usage_hours_per_day"]
        + noise
    )
    
    # Ensure physically bounded minimum consumption (> 2.0 kWh)
    df["simulated_daily_consumption_kwh"] = np.round(np.clip(calculated_kwh, 2.0, None), 2)
    
    # ---------------------------------------------------------
    # DERIVE TARGET CATEGORY
    # ---------------------------------------------------------
    def categorize(val: float) -> str:
        if val < LOW_THRESHOLD_KWH:
            return "Low"
        elif val <= HIGH_THRESHOLD_KWH:
            return "Medium"
        else:
            return "High"
            
    df["usage_category"] = df["simulated_daily_consumption_kwh"].apply(categorize)
    return df


def main():
    print("=" * 60)
    print("Generating Electricity Usage Dataset (Simulated)")
    print("=" * 60)
    
    df = generate_electricity_dataset()
    
    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[SUCCESS] Dataset saved to: {OUTPUT_FILE}")
    print(f"Total Records: {len(df)}")
    print(f"Features Generated: household_size, number_of_appliances, average_daily_usage_hours, ac_usage_hours_per_day")
    print(f"Target Label: usage_category (Low, Medium, High)")
    print(f"Intermediate Column: simulated_daily_consumption_kwh (Used ONLY for label derivation)")
    
    print("\nClass Distribution:")
    dist = df["usage_category"].value_counts()
    for cat in ["Low", "Medium", "High"]:
        cnt = dist.get(cat, 0)
        pct = (cnt / len(df)) * 100
        print(f"  - {cat:8s}: {cnt:4d} records ({pct:.1f}%)")
        
    print("\nFeature Summary Statistics:")
    print(df.describe().round(2).to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()
