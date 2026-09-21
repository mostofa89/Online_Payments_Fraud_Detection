Online Payment Fraud Detection

A machine-learning project for detecting fraudulent online payment transactions using a staged hybrid pipeline:

Logistic Regression → Random Forest → Neural Network (MLP)

This README is based on the complete workflow and results shown in the provided online_fraud full.pdf Google Colab export.

Overview

The project performs:

Dataset loading and inspection

Class-distribution analysis

Missing-value checking

Removal of account identifier columns

Median-based numerical missing-value handling

Balance-error feature engineering

Duplicate detection and removal

One-hot encoding of transaction type

Correlation heatmap analysis

Stratified 80/20 train-test split

Standard scaling of continuous features

Sequential hybrid model training

Confusion-matrix analysis

ROC curve and AUC calculation

Dataset

The notebook loads:

online_payments_fraud_detection_dataset (1).csv

from Google Drive.

Original dataset shape:

6,362,620 rows × 11 columns

The original columns are:

Column

Role

step

Time-step identifier

type

Transaction type

amount

Transaction amount

nameOrig

Origin account identifier

oldbalanceOrg

Origin balance before transaction

newbalanceOrig

Origin balance after transaction

nameDest

Destination account identifier

oldbalanceDest

Destination balance before transaction

newbalanceDest

Destination balance after transaction

isFraud

Binary fraud target

isFlaggedFraud

Existing fraud flag

Target Distribution

The notebook reports:

Class

Count

Legitimate (0)

6,354,407

Fraud (1)

8,213

This is a strongly imbalanced classification problem. class_weight='balanced' is therefore used in the Logistic Regression and Random Forest models.

Preprocessing

Remove account identifiers

The notebook drops:

df = df.drop(columns=['nameOrig'])
df = df.drop(columns=['nameDest'])

Missing values

Missing values are checked with df.isnull().sum(). The displayed data contains no missing values. The notebook also includes median imputation for numerical columns when missing values are present.

Balance-error features

Two engineered features are created:

df['errorBalanceOrig'] = (
    df['oldbalanceOrg'] + df['amount'] - df['newbalanceOrig']
)

df['errorBalanceDest'] = (
    df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
)

Duplicate removal

The notebook finds 543 duplicate rows, removes them, and then reports zero duplicates.

The resulting displayed dataset contains:

6,362,077 rows

One-hot encoding

type is encoded using OneHotEncoder(drop='first', sparse_output=False). The resulting transaction-type features shown in the notebook include:

type_CASH_OUT
type_DEBIT
type_PAYMENT
type_TRANSFER

After feature engineering and encoding, the displayed dataset has 6,362,077 rows × 14 columns.

Correlation Analysis

A correlation matrix is generated with:

corr = df.corr()

and visualized using a Seaborn heatmap.

Train/Test Split

The target is separated:

X = df.drop(columns=['isFraud'])
y = df['isFraud']

The notebook uses an 80/20 stratified split:

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

Scaling

The continuous columns scaled with StandardScaler are:

step
amount
oldbalanceOrg
newbalanceOrig
oldbalanceDest
newbalanceDest
errorBalanceOrig
errorBalanceDest

The scaler is fitted on training data and applied to the test data.

Hybrid Architecture

Transaction Features
        │
        ▼
Logistic Regression
        │
        │ predictions
        ▼
Original Features + LR Output
        │
        ▼
Random Forest
        │
        │ fraud probabilities
        ▼
Original Features + LR Output + RF Output
        │
        ▼
MLP Neural Network
        │
        ▼
Final Fraud Prediction

Model 1 — Logistic Regression

Configuration:

LogisticRegression(
    class_weight='balanced',
    max_iter=1000
)

Reported test metrics:

Metric

Score

Accuracy

0.9540

Precision

0.0265

Recall

0.9725

F1-Score

0.0517

The Logistic Regression predictions are then passed to the next stage as an additional feature.

Model 2 — Random Forest

The notebook adds the Logistic Regression predictions:

X_train_stage2['linreg_output'] = lr_train_pred
X_test_stage2['linreg_output'] = lr_test_pred

Random Forest configuration:

RandomForestClassifier(
    class_weight='balanced',
    random_state=42
)

Reported test metrics:

Metric

Score

Accuracy

0.9997

Precision

0.9795

Recall

0.7858

F1-Score

0.8720

Model 3 — MLP Neural Network

Random Forest fraud probabilities are added as another feature:

rf_train_prob = rf.predict_proba(X_train_stage2)[:, 1]
rf_test_prob = rf.predict_proba(X_test_stage2)[:, 1]

The final classifier is:

MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    random_state=42
)

Architecture:

Input
  ↓
128 neurons + ReLU
  ↓
64 neurons + ReLU
  ↓
32 neurons + ReLU
  ↓
Output

Final Hybrid Model Results

Reported test metrics:

Metric

Score

Accuracy

0.9997

Precision

0.9640

Recall

0.8164

F1-Score

0.8840

Confusion Matrix

The notebook reports:

[[1270727,     50],
 [     301,   1338]]

Interpreted using the notebook's Legit / Fraud labels:



Predicted Legit

Predicted Fraud

Actual Legit

1,270,727

50

Actual Fraud

301

1,338

The notebook visualizes this as Confusion Matrix — Hybrid Model.

Classification Report

Class

Precision

Recall

F1-Score

Legit

1.00

1.00

1.00

Fraud

0.96

0.82

0.88

The reported overall accuracy is 1.00 when rounded in the classification report.

ROC-AUC

The notebook calculates ROC data from the final MLP probabilities:

fpr, tpr, thresholds = roc_curve(y_test, mlp_prob)
roc_auc = auc(fpr, tpr)

The plot is titled:

ROC Curve - Hybrid Model (LR > RF > NN)

The numerical AUC output is not legible in the supplied PDF, so no value is invented here.

Model Comparison

Model

Accuracy

Precision

Recall

F1-Score

Logistic Regression

0.9540

0.0265

0.9725

0.0517

Random Forest

0.9997

0.9795

0.7858

0.8720

Hybrid MLP

0.9997

0.9640

0.8164

0.8840

These are the metrics displayed in the provided notebook export.

Technologies

Python

Google Colab

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

Scikit-learn components shown in the notebook include:

OneHotEncoder

StandardScaler

train_test_split

LogisticRegression

RandomForestClassifier

MLPClassifier

precision_score

recall_score

f1_score

accuracy_score

confusion_matrix

classification_report

roc_curve

auc

KMeans is imported in the notebook, but the supplied PDF does not show a completed K-Means experiment or clustering result.

Project Structure

Online-Fraud-Detection/
├── Online-Fraud-Detection.ipynb
├── README.md
├── requirements.txt
└── dataset/
    └── online_payments_fraud_detection_dataset (1).csv

The exact local repository structure can differ because the provided notebook loads the CSV from Google Drive.

Installation

pip install -r requirements.txt

Requirements

See requirements.txt for the libraries used by the notebook.

Running the Notebook

The notebook is designed for Google Colab and mounts Google Drive:

from google.colab import drive
drive.mount('/content/drive')

It then reads the CSV from:

/content/drive/MyDrive/AI ML Projeect/online_payments_fraud_detection_dataset (1).csv

Change this path when running the project elsewhere.

Reproducibility

The displayed experiment uses random_state=42 for the train/test split, Random Forest, and MLP. The same dataset and compatible software environment are required to reproduce the displayed results.

Future Extensions

The supplied PDF does not show these as completed experiments, but the project could later be extended with:

Hyperparameter tuning

Cross-validation

Precision-recall curve analysis

Threshold optimization

Feature-importance analysis

Probability calibration

Additional imbalance-handling methods

Additional classification models

Model persistence and inference API

Real-time fraud detection

Model monitoring and drift detection

Author

Mostofa Kamal Joy

Summary

The notebook demonstrates a staged fraud-detection pipeline:

Cleaning
  ↓
Feature Engineering
  ↓
Encoding
  ↓
Scaling
  ↓
Logistic Regression
  ↓
Random Forest
  ↓
MLP Neural Network
  ↓
Evaluation

The final reported hybrid model achieves 0.9997 accuracy, 0.9640 precision, 0.8164 recall, and 0.8840 F1-score on the displayed test set.
