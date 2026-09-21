# Execution Report — Online Payment Fraud Detection

**Project:** Online Payment Fraud Detection (Staged Hybrid Pipeline)
**Author:** Mostofa Kamal Joy
**Environment:** Google Colab
**Source:** `online_fraud full.pdf` (Colab export)

## 1. Objective

Execute a staged hybrid machine-learning pipeline — **Logistic Regression → Random Forest → MLP Neural Network** — to detect fraudulent transactions in a large, highly imbalanced online-payments dataset, and record the outcome of each execution stage.

## 2. Execution Environment

| Item | Detail |
|---|---|
| Platform | Google Colab |
| Data source | Google Drive (`drive.mount('/content/drive')`) |
| Input file | `online_payments_fraud_detection_dataset (1).csv` |
| Core libraries | NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn |

## 3. Execution Steps and Outcomes

### Step 1 — Data Loading
- Dataset loaded from Google Drive.
- **Result:** 6,362,620 rows × 11 columns loaded successfully.

### Step 2 — Class Distribution Check
- Target column `isFraud` inspected for class balance.
- **Result:** 6,354,407 legitimate transactions vs. 8,213 fraud transactions — a severe class imbalance (~0.13% fraud).

### Step 3 — Missing Value Check
- `df.isnull().sum()` executed across all columns.
- **Result:** No missing values found in the displayed run. Median-imputation logic included for numerical columns in case of future missing data.

### Step 4 — Column Removal
- Dropped `nameOrig` and `nameDest` (non-predictive identifier columns).
- **Result:** Executed without error; identifier columns removed from the working frame.

### Step 5 — Feature Engineering
- Computed `errorBalanceOrig` and `errorBalanceDest` to capture balance inconsistencies around each transaction.
- **Result:** Two new numerical features added successfully.

### Step 6 — Duplicate Removal
- Duplicate rows identified and dropped.
- **Result:** 543 duplicate rows found and removed. Dataset reduced to 6,362,077 rows; a re-check confirmed zero remaining duplicates.

### Step 7 — One-Hot Encoding
- `type` column encoded with `OneHotEncoder(drop='first', sparse_output=False)`.
- **Result:** Produced `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER`. Final feature set: 6,362,077 rows × 14 columns.

### Step 8 — Correlation Analysis
- Correlation matrix computed with `df.corr()` and rendered as a Seaborn heatmap.
- **Result:** Heatmap generated for exploratory review; no anomalies reported that blocked downstream steps.

### Step 9 — Train/Test Split
- Stratified 80/20 split executed (`random_state=42`).
- **Result:** Split completed with class proportions preserved in both partitions.

### Step 10 — Feature Scaling
- `StandardScaler` fitted on training data and applied to both partitions for 8 continuous columns (`step`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `errorBalanceOrig`, `errorBalanceDest`).
- **Result:** Scaling applied without leakage (fit on train, transform on both).

### Step 11 — Stage 1 Execution: Logistic Regression
- Configuration: `class_weight='balanced'`, `max_iter=1000`.
- **Result:**

| Metric | Score |
|---|---|
| Accuracy | 0.9540 |
| Precision | 0.0265 |
| Recall | 0.9725 |
| F1-Score | 0.0517 |

- High recall but very low precision, consistent with a deliberately loose first-stage filter. Predictions passed forward as an added feature.

### Step 12 — Stage 2 Execution: Random Forest
- Logistic Regression predictions appended as `linreg_output`.
- Configuration: `class_weight='balanced'`, `random_state=42`.
- **Result:**

| Metric | Score |
|---|---|
| Accuracy | 0.9997 |
| Precision | 0.9795 |
| Recall | 0.7858 |
| F1-Score | 0.8720 |

- Substantial precision gain over Stage 1; fraud probabilities generated for Stage 3.

### Step 13 — Stage 3 Execution: MLP Neural Network
- Random Forest fraud probabilities appended as an additional feature.
- Configuration: `hidden_layer_sizes=(128, 64, 32)`, `activation='relu'`, `random_state=42`.
- **Result (final hybrid model):**

| Metric | Score |
|---|---|
| Accuracy | 0.9997 |
| Precision | 0.9640 |
| Recall | 0.8164 |
| F1-Score | 0.8840 |

### Step 14 — Confusion Matrix
- Computed on the final MLP output.

```
[[1270727,     50],
 [     301,   1338]]
```

| | Predicted Legit | Predicted Fraud |
|---|---|---|
| **Actual Legit** | 1,270,727 | 50 |
| **Actual Fraud** | 301 | 1,338 |

### Step 15 — Classification Report

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Legit | 1.00 | 1.00 | 1.00 |
| Fraud | 0.96 | 0.82 | 0.88 |

### Step 16 — ROC Curve / AUC
- `roc_curve` and `auc` computed from MLP probabilities and plotted as *"ROC Curve - Hybrid Model (LR > RF > NN)"*.
- **Result:** Curve generated successfully; the numeric AUC value was not legible in the source export and is not reported here to avoid inventing a figure.

### Step 17 — K-Means (Imported, Not Executed)
- `KMeans` was imported in the notebook.
- **Result:** No completed clustering run or output appears in the supplied export; this step is not part of the executed pipeline.

## 4. Summary of Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.9540 | 0.0265 | 0.9725 | 0.0517 |
| Random Forest | 0.9997 | 0.9795 | 0.7858 | 0.8720 |
| Hybrid (LR → RF → MLP) | 0.9997 | 0.9640 | 0.8164 | 0.8840 |

## 5. Observations

- Each stage of the pipeline executed in sequence without errors, with outputs from one stage correctly feeding into the next as engineered features.
- The staged design trades some of Logistic Regression's raw recall for a large gain in precision at the Random Forest and MLP stages, producing a final model with 0.9640 precision and 0.8164 recall — a reasonable balance for a fraud-detection use case where false positives carry a real review cost.
- The pipeline handled a 6.3M-row, heavily imbalanced dataset (~0.13% fraud) end-to-end, including cleaning, feature engineering, encoding, scaling, three sequential model fits, and full evaluation (confusion matrix, classification report, ROC/AUC).
- K-Means was imported but not exercised in this run, and the AUC value could not be confirmed from the source export — both are flagged above rather than filled in with assumed numbers.

## 6. Conclusion

The staged hybrid pipeline executed successfully end-to-end on the full dataset, with each of the three models (Logistic Regression, Random Forest, MLP) running as intended and passing its output forward to the next stage. The final hybrid model achieved **0.9997 accuracy, 0.9640 precision, 0.8164 recall, and 0.8840 F1-score** on the held-out test set, confirming that the sequential architecture improved fraud precision substantially over the first-stage Logistic Regression model while retaining strong recall.
