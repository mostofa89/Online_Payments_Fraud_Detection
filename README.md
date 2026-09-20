# Online Payment Fraud Detection

A machine-learning project for detecting fraudulent online financial transactions using a hybrid supervised-learning pipeline and unsupervised clustering.

## Project Overview

This project uses the **PaySim-style online payment fraud dataset** and explores:

1. Data loading and exploratory analysis
2. Data cleaning and duplicate removal
3. Balance-error feature engineering
4. One-hot encoding of transaction type
5. Feature scaling
6. Supervised fraud detection using:
   - Logistic Regression
   - Random Forest
   - Multilayer Perceptron (Neural Network)
7. A hybrid pipeline:

```text
Input Features
      |
      v
Logistic Regression
      |
      | prediction output
      v
Random Forest
      |
      | fraud probability
      v
Neural Network (MLP)
      |
      v
Final Fraud Prediction
```

8. Unsupervised learning with K-Means clustering
9. Evaluation using classification metrics, confusion matrix, ROC curve, and AUC

> **Important:** The supplied PDF is a printed Google Colab notebook, not the original `.ipynb` file. The PDF contains screenshots of code and outputs. Therefore, the exact original notebook cannot be reproduced perfectly from the PDF alone.

## Dataset

The notebook uses a CSV file named:

```text
online_payments_fraud_detection_dataset (1).csv
```

The dataset is expected to contain these columns:

- `step`
- `type`
- `amount`
- `nameOrig`
- `oldbalanceOrg`
- `newbalanceOrig`
- `nameDest`
- `oldbalanceDest`
- `newbalanceDest`
- `isFraud`
- `isFlaggedFraud`

The target column is:

```text
isFraud
```

where:

- `0` = legitimate transaction
- `1` = fraudulent transaction

The PDF reports an initial dataset size of **6,362,620 rows and 11 columns**. It also reports:

- Legitimate transactions: `6,354,407`
- Fraudulent transactions: `8,213`
- Duplicate rows detected: `543`
- Missing values: `0` in the displayed data

The target is highly imbalanced, so accuracy should not be used as the only evaluation metric.

## Features Created

Two balance consistency features are created:

```python
df["errorBalanceOrig"] = (
    df["oldbalanceOrg"] + df["amount"] - df["newbalanceOrig"]
)

df["errorBalanceDest"] = (
    df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
)
```

The identifier columns `nameOrig` and `nameDest` are removed.

The categorical `type` column is encoded using one-hot encoding with `drop="first"`.

The continuous columns considered for scaling are:

```python
continuous_cols = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
]
```

## Models

### 1. Logistic Regression

The notebook uses class balancing:

```python
LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)
```

The PDF reports the following Logistic Regression test metrics:

| Metric | Reported value |
|---|---:|
| Accuracy | 0.9548 |
| Precision | 0.6265 |
| Recall | 0.9725 |
| F1-score | 0.7517 |

These values are transcribed from the PDF screenshot and should be revalidated by running the original notebook or a corrected implementation.

### 2. Random Forest

The Random Forest receives the original processed features plus the Logistic Regression prediction:

```python
X_train_stage2["lr_output"] = lr_train_pred
X_test_stage2["lr_output"] = lr_test_pred
```

The PDF shows the Random Forest stage, but its complete numerical output is not clearly readable in the supplied screenshots.

### 3. Neural Network

The final stage uses an MLP:

```python
MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    random_state=42
)
```

The MLP receives:

- Processed input features
- Logistic Regression output
- Random Forest fraud probability

The PDF includes code for accuracy, precision, recall, F1-score, confusion matrix, classification report, and ROC-AUC evaluation. However, the complete numerical results are not clearly visible in the PDF.

### 4. K-Means Clustering

The notebook also experiments with K-Means clustering:

```python
KMeans(
    n_clusters=3,
    random_state=1,
    n_init=10
)
```

An elbow-method experiment evaluates values of `k` from `1` through `9` using SSE/inertia.

K-Means is exploratory in this project. Cluster IDs do not inherently represent fraud labels, so a cluster-to-label mapping must be defined before treating clustering as a classifier.

## Installation

Create and activate a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Setup

Create a data directory:

```text
data/
└── online_payments_fraud_detection_dataset.csv
```

Do not commit large datasets to GitHub unless the dataset license and repository policy permit it.

Update the dataset path in the notebook or script:

```python
DATA_PATH = "data/online_payments_fraud_detection_dataset.csv"
```

## Recommended Execution Order

1. Load the dataset.
2. Inspect shape, data types, descriptive statistics, and class distribution.
3. Check missing values.
4. Remove identifier columns.
5. Create balance-error features.
6. Remove duplicate rows.
7. Encode `type`.
8. Split the dataset using stratification.
9. Fit the scaler on the training set only.
10. Transform the test set using the fitted scaler.
11. Train Logistic Regression.
12. Add Logistic Regression predictions to the feature set.
13. Train Random Forest.
14. Add Random Forest probabilities to the feature set.
15. Train the MLP.
16. Evaluate the final model.
17. Run K-Means separately as an exploratory experiment.

## Evaluation Metrics

Because fraud is a minority class, report:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC
- Preferably PR-AUC / Average Precision
- False-positive rate
- False-negative rate

For fraud detection, recall and false negatives are particularly important, but increasing recall may also increase false positives.

## Important Implementation Notes

The PDF contains several transcription-visible or notebook-code issues that should be corrected before execution:

- `accuracy_score` is used but is not shown in the import list.
- Some variable names appear inconsistent, such as `lr` versus `1r`, and `rf` versus `orf`.
- Some screenshots show malformed or incomplete code caused by PDF rendering.
- The confusion-matrix plotting cell appears incomplete in the PDF.
- The ROC label should be an f-string if `roc_auc` is interpolated:

```python
plt.plot(
    fpr,
    tpr,
    label=f"Hybrid Model (AUC = {roc_auc:.3f})"
)
```

- Scaling should be performed after the train/test split, fitting only on training data.
- The K-Means experiment should use a clearly defined feature matrix and should not reuse supervised-learning variables accidentally.
- A production-ready implementation should use a `Pipeline` and preserve preprocessing/model artifacts.

## Limitations

- The project is evaluated on a highly imbalanced dataset.
- The dataset is simulated and may not represent every real-world fraud pattern.
- The hybrid model uses predictions from earlier models as later-stage features; this can cause data leakage if out-of-fold predictions are not used during training.
- The notebook's displayed code is incomplete in some places.
- No independent external test set is shown in the PDF.
- Zero-day or previously unseen fraud detection is not demonstrated.

## Suggested Repository Structure

```text
online-fraud-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   └── online_fraud_detection.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── reports/
│   └── execution_report.md
└── outputs/
    ├── figures/
    └── models/
```

## Reproducibility

Use fixed random states where applicable:

- Train/test split: `random_state=42`
- Random Forest: `random_state=42`
- MLP: `random_state=42`
- K-Means: `random_state=1`

For a reproducible research implementation, save:

- Preprocessing configuration
- Feature names
- Fitted scaler
- Trained models
- Threshold used for fraud classification
- Test metrics
- Confusion matrix
- ROC and precision-recall curves

## Project Status

**Status: Notebook review completed; full execution pending.**

The supplied PDF was inspected, but the original CSV dataset and executable notebook were not included. Consequently, the entire pipeline could not be executed and independently verified from the PDF alone. The metrics explicitly marked as reported above come from the PDF screenshots.
