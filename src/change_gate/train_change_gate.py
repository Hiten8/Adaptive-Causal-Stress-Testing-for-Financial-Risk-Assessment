"""
Train the Confidence-Guided Adaptive Change Gate.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

from xgboost import XGBClassifier

from src.storage.causal_state_repository import (
    CausalStateRepository
)

from src.change_gate.feature_extractor import (
    build_training_sample
)

MODEL_FOLDER = Path("models/change_gate")
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

RESULT_FOLDER = Path("results")
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

DATASET_FILE = RESULT_FOLDER / "change_gate_training_dataset.csv"
PREDICTION_FILE = RESULT_FOLDER / "change_gate_predictions.csv"

# Load Causal States
repository = CausalStateRepository()
states = repository.load_all()

print("=" * 70)
print("Training Confidence-Guided Adaptive Change Gate")
print("=" * 70)
print(f"Causal States Loaded : {len(states)}")


# Build Dataset
dataset = []
for i in range(1, len(states)):
    sample = build_training_sample(
        states[i - 1],
        states[i]
    )

    dataset.append(sample)

dataset = pd.DataFrame(dataset)

dataset.to_csv(
    DATASET_FILE,
    index=False
)

print(f"Training Samples : {len(dataset)}")

# Train/Test Split
X = dataset.drop(columns=["UpdateLabel"])
y = dataset["UpdateLabel"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train XGBoost
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.80,
    colsample_bytree=0.80,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# Predictions
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.50).astype(int)

# Metrics
accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    probabilities
)

cm = confusion_matrix(
    y_test,
    predictions
)

print()
print("=" * 70)
print("Evaluation")
print("=" * 70)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {auc:.4f}")
print()
print("Confusion Matrix")
print(cm)

# Save Predictions
prediction_df = X_test.copy()
prediction_df["TrueLabel"] = y_test.values
prediction_df["Prediction"] = predictions
prediction_df["StructuralChangeScore"] = probabilities
prediction_df.to_csv(
    PREDICTION_FILE,
    index=False
)

# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance.to_csv(
    MODEL_FOLDER / "feature_importance.csv",
    index=False
)

# Save Model
joblib.dump(
    model,
    MODEL_FOLDER / "change_gate_xgboost.pkl"
)

print()
print("=" * 70)
print("Training Completed Successfully")
print("=" * 70)
print()
print("Saved:")
print(f"Dataset          : {DATASET_FILE}")
print(f"Predictions      : {PREDICTION_FILE}")
print(f"Model            : {MODEL_FOLDER/'change_gate_xgboost.pkl'}")
print(f"Feature Importance: {MODEL_FOLDER/'feature_importance.csv'}")
print()
print("Top 10 Important Features")
print(importance.head(10))