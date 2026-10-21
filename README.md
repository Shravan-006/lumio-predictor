# ⚡ Electricity Usage Category Predictor Using Machine Learning

A complete, production-ready, local Machine Learning web application designed to predict household electricity consumption categories (**Low**, **Medium**, **High**) from everyday lifestyle features. Built with **Python**, **scikit-learn**, and **Streamlit** for academic demonstration at college project expos.

---

## 🌟 Project Overview

Domestic energy consumers often lack immediate visibility into whether their consumption trends place them in low, moderate, or high utility tariff bands until their monthly or bi-monthly utility bill arrives.

This application provides an **instant, decision-tree-powered classification tool** that evaluates four key household attributes:
1. **Household Size** (Number of residents)
2. **Number of Appliances** (Total electrical appliances owned)
3. **Average Daily Usage Hours** (Active operating hours)
4. **Daily Air Conditioner (AC) Usage Hours** (High-drain appliance operating time)

The application predicts the consumption category and delivers targeted, practical energy-saving strategies to help users optimize electricity usage.

> [!IMPORTANT]
> **Educational & Synthetic Data Disclosure:**  
> This project operates on a synthetically generated dataset based on physical residential energy principles and Gaussian variance. It is intended for educational and academic demonstration purposes and cannot replace an industrial, on-site smart meter energy audit.

---

## 🏗️ Project Architecture & Folder Structure

```
electricity_usage_predictor/
│
├── app.py                      # Main Streamlit web application dashboard
├── generate_dataset.py         # Reproducible synthetic dataset generator
├── train_model.py              # ML training, evaluation, and artifact export script
├── requirements.txt            # Python package dependencies
├── README.md                   # Complete documentation and setup guide
├── VIVA_NOTES.md               # Comprehensive viva preparation notes & FAQs
│
├── electricity_usage.csv       # Generated dataset (800 simulated household records)
│
├── models/
│   ├── electricity_model.joblib # Serialized scikit-learn DecisionTreeClassifier
│   └── model_metadata.json     # Hyperparameters, feature schema, and test metrics
│
└── reports/
    └── evaluation_results.txt  # Human-readable evaluation report and confusion matrix
```

---

## 💻 Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.10+ / 3.13 | Core development language |
| **User Interface** | Streamlit | Responsive, interactive local web UI |
| **Data Processing** | Pandas, NumPy | Dataset handling, vector operations, array math |
| **Machine Learning** | scikit-learn | DecisionTreeClassifier (CART algorithm) |
| **Model Serialization**| Joblib | Fast binary model persistence and inference loading |
| **Data Visualization** | Plotly Express & Graph Objects | Interactive charts, confusion matrix heatmaps |
| **Static Reporting** | Matplotlib / Text Reports | Text and tabular evaluation outputs |

---

## 📊 Dataset Specifications

The dataset contains **800 simulated household records** generated with physical energy consumption dynamics.

### Feature Definitions

| Column Name | Type | Range | Description |
|---|---|---|---|
| `household_size` | Integer | 1 – 8 | Number of regular occupants in the household. |
| `number_of_appliances` | Integer | 3 – 25 | Total count of electrical appliances (fans, lights, fridge, TV, washing machine, etc.). |
| `average_daily_usage_hours` | Float | 4.0 – 18.0 | Estimated cumulative hours appliances are operational each day. |
| `ac_usage_hours_per_day` | Float | 0.0 – 14.0 | Daily hours the air conditioner is turned on ($\le$ total usage hours). |
| `simulated_daily_consumption_kwh` | Float | 2.5 – 45.0 | **Intermediate calculation only** used to derive the ground-truth category. **Never fed into the model as an input feature.** |
| `usage_category` | String | Low, Medium, High | **Target Label** for supervised classification. |

### Classification Thresholds
- **Low Demand:** $< 10.0\text{ kWh/day}$ (Approx. $< 300\text{ units/month}$)
- **Medium Demand:** $10.0\text{ to } 22.0\text{ kWh/day}$ (Approx. $300\text{ to } 660\text{ units/month}$)
- **High Demand:** $> 22.0\text{ kWh/day}$ (Approx. $> 660\text{ units/month}$)

### Ground Truth Generation Formula
$$\text{Consumption (kWh)} = (1.25 \times \text{Size}) + \left(0.38 \times \text{Appliances} \times \frac{\text{Usage Hours}}{8.0}\right) + (1.45 \times \text{AC Hours}) + \mathcal{N}(0, 1.2^2)$$

---

## 🧠 Machine Learning Model Details

- **Algorithm:** Decision Tree Classifier (`sklearn.tree.DecisionTreeClassifier`)
- **Splitting Criterion:** Gini Impurity ($Gini = 1 - \sum p_i^2$)
- **Train/Test Split:** 80% Training (640 records), 20% Testing (160 records), Stratified
- **Regularization:** `max_depth=4`, `min_samples_split=8`, `min_samples_leaf=4`
- **Why Decision Tree?**
  - **Transparent and Explainable:** Each split corresponds to a logical threshold (e.g., $AC \le 3.5\text{ hrs}$).
  - **Scale Invariant:** Does not require standard scaling or normalization.
  - **Fast Inference:** Real-time predictions in $< 2\text{ ms}$ on any laptop CPU.

---

## 🚀 Setup and Execution Instructions (Windows)

### 1. Open PowerShell in Project Directory
Navigate to the project folder:
```powershell
cd "c:\Users\Shravan\Downloads\ML project"
```

### 2. Create and Activate Virtual Environment
```powershell
python -m venv .venv
```
If PowerShell restricts script execution, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```
*(This sets the policy only for the current PowerShell terminal session without altering system-wide security).*

### 3. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 4. Generate Dataset
```powershell
python generate_dataset.py
```
This generates `electricity_usage.csv` with 800 reproducible records.

### 5. Train Model & Export Evaluation Artifacts
```powershell
python train_model.py
```
This trains the Decision Tree, writes `models/electricity_model.joblib`, `models/model_metadata.json`, and `reports/evaluation_results.txt`.

### 6. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Streamlit will launch locally in your browser at:
`http://localhost:8501`

---

## 🎪 Expo Demonstration Guide (Faculty Walkthrough)

To present the project smoothly during the college exhibition, follow this 5-step sequence:

1. **Dashboard (Step 1):**
   - Point out the 4 KPI cards: Total records (800), Input features (4), Categories (3), and Test Accuracy.
   - Show the **Category Distribution** donut chart and the **Daily Consumption Spread** box plot.
   - Explain the system workflow diagram.
2. **Predict Usage (Step 2):**
   - Enter typical household values (e.g., 4 members, 12 appliances, 8 hours usage, 3.5 AC hours).
   - Click **Predict Usage Category**.
   - Show the result (e.g., **Medium** or **High**), the probability breakdown across classes, and the tailored energy-saving suggestions.
   - Test an edge case: Show how the validator warns if AC hours exceed daily usage hours.
3. **Model Evaluation (Step 3):**
   - Review the held-out test accuracy, macro precision, recall, and F1-score.
   - Walk the examiner through the **Confusion Matrix Heatmap** (diagonal values indicate correct predictions).
   - Display the **Gini Feature Importance** chart: explain why AC usage is the primary driver of electricity category.
4. **About Project (Step 4):**
   - Show the problem statement, CART formula, and physical generation equation.
   - Conclude by discussing limitations and future scope (IoT smart meters, time-series forecasting).

---

## 🔧 Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| `ExecutionPolicy` error in PowerShell | Windows default security script policy | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then run `.venv\Scripts\Activate.ps1`. |
| Missing dataset or model error on app launch | `generate_dataset.py` or `train_model.py` not run | Run `python generate_dataset.py` followed by `python train_model.py`. |
| Port 8501 is already in use | Another Streamlit instance is running | Run `streamlit run app.py --server.port 8502`. |
| Package import errors | Running outside the virtual environment | Ensure `(.venv)` appears in your command prompt, or run `.venv\Scripts\python app.py`. |

---

## 📜 Limitations & Future Improvements

- **Dataset Nature:** Built on synthetic data to demonstrate machine learning fundamentals. Real deployment requires IoT smart meter telemetry.
- **Seasonality:** Real-world consumption varies heavily between winter and summer; dynamic seasonal factors can be added.
- **Future Scope:**
  - Hardware integration with ESP32 / PZEM-004T energy sensors over MQTT.
  - Time-series consumption forecasting using ARIMA or LSTM.
  - Multi-tariff billing engines for local electricity boards.
