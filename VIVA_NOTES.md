# 🎓 Viva Preparation Notes: Electricity Usage Category Predictor

This document is your complete cheat-sheet for faculty questions, external examiner evaluations, and project expo presentations. Keep these answers concise, confident, and technically accurate.

---

## 1. 30-Second Elevator Pitch
> *"Good morning, sir/madam. My project is an 'Electricity Usage Category Predictor'. Most households only find out their electricity consumption tier when they receive their monthly electricity bill. Our system uses a supervised Decision Tree Classifier to predict whether a home falls into Low, Medium, or High consumption categories based on four everyday household inputs: family size, appliance count, daily usage hours, and AC usage hours. The application is deployed locally via a Streamlit dashboard that delivers instant categorization, confidence scores, and customized energy conservation strategies."*

---

## 2. 1-Minute Comprehensive Explanation
> *"Domestic electricity usage often spikes unpredictably due to high-consumption appliances like air conditioners. To address this, we developed an educational machine learning application that classifies residential energy demand.
>
> We generated a physically motivated synthetic dataset of 800 household records incorporating base standby power, appliance counts, and air conditioner operational hours with Gaussian variance. Using an 80/20 stratified split, we trained a scikit-learn Decision Tree Classifier constrained to depth 4 to prevent overfitting. 
>
> The model achieves strong classification accuracy and high F1-score on held-out test data. The trained pipeline is serialized with Joblib and served through an interactive Streamlit web interface. Users can input their household details to receive immediate categorization, class probabilities, and practical energy-saving advice. Crucially, we treat this as an educational prototype and transparently disclose the synthetic nature of the dataset."*

---

## 3. Why Decision Tree Classifier?
1. **Explainability & Transparency:** Unlike 'black-box' models (Neural Networks, SVMs), Decision Trees produce human-interpretable rules ($IF \dots THEN \dots$).
2. **Handles Non-Linear Thresholds:** Captures logical energy interactions (e.g., high appliance count + high AC usage leads to High tier).
3. **No Feature Scaling Required:** Invariant to differences in units (occupants vs. hours).
4. **Fast and Lightweight:** Runs inferences in less than 2 milliseconds with near-zero memory footprint.

---

## 4. Fundamental ML Concepts in This Project

### What is Supervised Learning?
Supervised learning is a machine learning paradigm where the algorithm learns a mapping function from input features ($X$) to known ground-truth labels ($y$) using labeled training data.

### Features vs. Target
- **Features ($X$):** The 4 input variables:
  1. `household_size`
  2. `number_of_appliances`
  3. `average_daily_usage_hours`
  4. `ac_usage_hours_per_day`
- **Target ($y$):** The output label to predict: `usage_category` (`Low`, `Medium`, `High`).

### Why Separate Training and Testing Sets?
To evaluate the model's **generalization capability** on unseen data. If we evaluated on the training data, the model could simply memorize the inputs (overfitting), giving a false impression of predictive capability. We used an **80% training / 20% testing stratified split**.

### What is Stratification?
Stratification ensures that the proportion of classes (`Low`, `Medium`, `High`) is identical in both the training set and the test set, preventing class imbalance bias during training.

### What is Overfitting and How Was It Prevented?
Overfitting occurs when a model learns the random noise and idiosyncrasies of the training data rather than the underlying pattern, resulting in poor test accuracy.
**Prevention in our project:**
- Constrained `max_depth=4` (preventing an overly deep, complex tree).
- Set `min_samples_split=8` and `min_samples_leaf=4` (requiring multiple samples to create or maintain a branch).

---

## 5. Evaluation Metrics Explained

| Metric | Simple Explanation | Formula |
|---|---|---|
| **Accuracy** | Total correct predictions divided by total predictions. | $\frac{TP + TN}{Total}$ |
| **Precision** | Out of all instances predicted as category $C$, how many were actually $C$? (Measures reliability of positive calls). | $\frac{TP}{TP + FP}$ |
| **Recall (Sensitivity)** | Out of all actual instances of category $C$, how many did the model identify? (Measures ability to find all positives). | $\frac{TP}{TP + FN}$ |
| **F1-Score** | Harmonic mean of Precision and Recall. Balances precision and recall when classes may not be perfectly symmetrical. | $2 \times \frac{P \times R}{P + R}$ |
| **Confusion Matrix** | A table showing Actual vs. Predicted classifications. Diagonal cells represent correct predictions; off-diagonal cells represent misclassifications. | 2D Matrix of counts |

---

## 6. The Synthetic Data Question (Very Important!)

### Why Did You Use Synthetic Data?
Real smart-meter household datasets at the appliance-level often involve strict consumer privacy regulations, high licensing costs, or lack labeled lifestyle features (like resident count and AC operating hours). We generated simulated data based on domain-grounded physical electricity formulas to model the workflow end-to-end.

### Why Does Synthetic Data Limit Real-World Conclusions?
Synthetic data reflects the exact mathematical assumptions and random distributions programmed into the generator. Real households have unpredictable behaviors, unmonitored power leaks, voltage fluctuations, and differing appliance star ratings. Therefore, this model validates the **ML pipeline architecture**, but cannot claim field accuracy on real utility meters without retraining on physical hardware sensor data.

---

## 7. How Does Streamlit Use the Saved Model?
1. The model is trained in `train_model.py` and saved as a binary file: `models/electricity_model.joblib`.
2. In `app.py`, `joblib.load()` loads the pre-fitted model into memory using Streamlit's `@st.cache_resource` decorator so it is only loaded once.
3. When the user fills the form and clicks "Predict", their inputs are converted into a Pandas DataFrame with the exact feature column names and order.
4. `model.predict(input_df)` returns the predicted class (`Low`, `Medium`, or `High`), while `model.predict_proba(input_df)` returns confidence probabilities.

---

## 8. 20 Likely Viva Questions & Instant Answers

**Q1: What is the main objective of your project?**  
*Ans:* To classify household electricity consumption into Low, Medium, or High categories using a Decision Tree Classifier based on lifestyle parameters.

**Q2: What is the target variable and what type of problem is this?**  
*Ans:* The target variable is `usage_category` (`Low`, `Medium`, `High`). It is a multi-class supervised classification problem.

**Q3: Why not predict the exact numerical kWh instead of categories?**  
*Ans:* Exact numerical consumption varies minute-by-minute with weather and voltage. For consumer decision-making and utility tariff planning, knowing the tariff slab category (Low/Medium/High) is more actionable and robust.

**Q4: Which splitting criterion did you use for the Decision Tree?**  
*Ans:* We used the **Gini Impurity** criterion, which measures how often a randomly chosen element from the set would be incorrectly labeled.

**Q5: What is the range of Gini Impurity?**  
*Ans:* For multi-class classification, Gini ranges from 0.0 (pure node, all samples belong to one class) to $1 - \frac{1}{C}$ (maximum impurity, equally mixed classes).

**Q6: What is the difference between Gini Impurity and Entropy?**  
*Ans:* Gini uses squared probabilities ($1 - \sum p_i^2$), while Entropy uses logarithms ($-\sum p_i \log_2 p_i$). Gini is computationally faster because it avoids logarithmic calculations. Both yield very similar trees.

**Q7: What is pruning in Decision Trees?**  
*Ans:* Pruning is the technique of reducing the size of a decision tree by removing sections that provide little predictive power, in order to combat overfitting. Pre-pruning was implemented via `max_depth=4` and `min_samples_leaf=4`.

**Q8: What are your four input features?**  
*Ans:* `household_size`, `number_of_appliances`, `average_daily_usage_hours`, and `ac_usage_hours_per_day`.

**Q9: Which feature has the highest importance in your model?**  
*Ans:* `ac_usage_hours_per_day`, because air conditioners typically draw 1.2 to 1.8 kW of power, making them the single largest contributor to residential electricity consumption.

**Q10: Why did you not include `simulated_daily_consumption_kwh` as a feature?**  
*Ans:* That would cause **data leakage**. `simulated_daily_consumption_kwh` was used to create the target labels during data generation; feeding it to the model would give away the answer and result in an artificial, invalid model.

**Q11: What is data leakage?**  
*Ans:* When information from outside the training dataset (or target label information) is inadvertently used to train the model, producing unrealistically optimistic performance that fails completely in the real world.

**Q12: Why did you use `joblib` instead of Python `pickle`?**  
*Ans:* `joblib` is optimized for serializing large NumPy arrays and scikit-learn models efficiently, reducing memory footprint and serialization time.

**Q13: How did you validate that AC hours do not exceed total appliance hours?**  
*Ans:* In both dataset generation and the Streamlit UI, an explicit validation rule ensures `ac_usage_hours_per_day <= average_daily_usage_hours`. If violated, the UI displays a validation warning.

**Q14: What is the difference between Macro and Weighted F1-Score?**  
*Ans:* Macro F1 calculates the F1-score for each class independently and averages them with equal weight. Weighted F1 weights each class's F1-score by its proportion (support) in the dataset.

**Q15: What is the advantage of Streamlit over Flask or Django for this project?**  
*Ans:* Streamlit allows rapid prototyping of interactive, data-driven web dashboards entirely in Python without writing separate HTML, CSS, JavaScript, or REST API controllers.

**Q16: Can a Decision Tree handle categorical features directly in scikit-learn?**  
*Ans:* In scikit-learn, all inputs to `DecisionTreeClassifier` must be numerical. If categorical features exist, they must be encoded (e.g., One-Hot Encoding or Ordinal Encoding). All our features are continuous or discrete numeric.

**Q17: What is the time complexity of predicting with a Decision Tree?**  
*Ans:* The prediction time complexity is $\mathcal{O}(d)$, where $d$ is the tree depth. Since our `max_depth` is 4, each prediction takes at most 4 comparisons, making inference nearly instantaneous.

**Q18: What is the significance of the `random_state` parameter?**  
*Ans:* Setting `random_state=42` ensures deterministic, reproducible results across dataset generation, train/test splitting, and tree node splitting.

**Q19: If you had real IoT data, how would the architecture change?**  
*Ans:* We would replace the CSV file with an MQTT broker or time-series database (like InfluxDB), gathering live readings from ESP32/CT current sensors, and feed sliding window averages into the classifier.

**Q20: What are the main limitations of your project?**  
*Ans:* Synthetic data, lack of seasonal temperature/weather variables, and absence of real-time multi-slab utility tariff integration.

---

## 9. Critical Code Lines to Understand

1. **Splitting the Data:**
   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.20, random_state=42, stratify=y
   )
   ```
   *Explains: 80/20 held-out evaluation with stratified class preservation.*

2. **Decision Tree with Regularization:**
   ```python
   clf = DecisionTreeClassifier(
       criterion="gini", max_depth=4, min_samples_split=8, min_samples_leaf=4, random_state=42
   )
   clf.fit(X_train, y_train)
   ```
   *Explains: Fitting the tree with controlled depth to prevent overfitting.*

3. **Loading and Running Prediction in Streamlit:**
   ```python
   model = joblib.load("models/electricity_model.joblib")
   prediction = model.predict(input_df)[0]
   probabilities = model.predict_proba(input_df)[0]
   ```
   *Explains: Fast model inference and confidence calculation.*

---

## 10. Top 10 Expo Presentation Rules
1. **Be Honest About Synthetic Data:** State upfront that the dataset is simulated for academic demonstration. Examiners appreciate scientific integrity.
2. **Highlight the Problem:** Emphasize that surprise utility bills are a real problem that early categorization solves.
3. **Show, Don't Just Tell:** Demonstrate the UI live. Adjust AC usage sliders to show the class change from Low to High.
4. **Demonstrate Input Validation:** Move the AC slider above total usage hours to show that your app catches user input errors gracefully.
5. **Show the Confusion Matrix:** Point to the diagonal cells on the Model Evaluation page to prove test set accuracy.
6. **Explain Feature Importance:** Show that AC hours dominate the Gini importance, reflecting real-world thermodynamics.
7. **Discuss Energy Conservation:** Show how the app provides tailored tips (thermostat at 24°C saves ~6% per degree).
8. **Know Your Hyperparameters:** Remember `max_depth=4`, `random_state=42`, and `80/20 split`.
9. **Keep Answers Concise:** Answer directly, then stop. Do not over-explain unless prompted.
10. **Discuss Future Scope Confidently:** Mention IoT smart sensors and LSTM time-series forecasting.
