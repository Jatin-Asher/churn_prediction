import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from src.utils import save_object


class ModelTrainer:

    def __init__(self):

        self.trained_model_path = os.path.join(
            "artifacts",
            "model.pkl"
        )

    def initiate_model_training(
        self,
        train_path,
        test_path
    ):

        print("Loading transformed datasets...")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        TARGET_COLUMN = "Churn Value"

        X_train = train_df.drop(
            columns=[TARGET_COLUMN]
        )

        y_train = train_df[TARGET_COLUMN]

        X_test = test_df.drop(
            columns=[TARGET_COLUMN]
        )

        y_test = test_df[TARGET_COLUMN]

        print(f"X_train Shape: {X_train.shape}")
        print(f"X_test Shape : {X_test.shape}")

        print("\nTraining Random Forest Model...")

        model = RandomForestClassifier(
            n_estimators=100,
            min_samples_split=10,
            min_samples_leaf=4,
            max_depth=10,
            class_weight="balanced",
            random_state=42
        )

        model.fit(
            X_train,
            y_train
        )

        print("Model training completed.")

        y_pred = model.predict(
            X_test
        )

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred
        )

        recall = recall_score(
            y_test,
            y_pred
        )

        f1 = f1_score(
            y_test,
            y_pred
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )

        print("\n===== MODEL PERFORMANCE =====")

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")

        print("\nClassification Report:\n")

        print(
            classification_report(
                y_test,
                y_pred
            )
        )

        print("\nConfusion Matrix:\n")

        print(
            confusion_matrix(
                y_test,
                y_pred
            )
        )

        save_object(
            self.trained_model_path,
            model
        )

        print(
            f"\nModel saved successfully at: {self.trained_model_path}"
        )

        return model