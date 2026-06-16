import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import pandas as pd
import sys
from pathlib import Path

@dataclass
class DataTransformationConfig:

    PROJECT_ROOT = os.getcwd()

    interim_data_path = os.path.join(
        PROJECT_ROOT,
        "dataset",
        "interim",
        "customer_churn_interim.csv"
    )

    train_processed_path = os.path.join(
        PROJECT_ROOT,
        "dataset",
        "processed",
        "train_processed.csv"
    )

    test_processed_path = os.path.join(
        PROJECT_ROOT,
        "dataset",
        "processed",
        "test_processed.csv"
    )

    preprocessor_path = os.path.join(
        PROJECT_ROOT,
        "artifacts",
        "preprocessor.pkl"
    )

class DataTransformation:

    def __init__(self):

        self.config = DataTransformationConfig()

    def get_preprocessor(self):

        try:

            numerical_columns = [
                "Tenure Months",
                "Monthly Charges",
                "Total Charges",
                "CLTV"
            ]

            categorical_columns = [
                "Gender",
                "Partner",
                "Dependents",
                "Phone Service",
                "Multiple Lines",
                "Internet Service",
                "Online Security",
                "Online Backup",
                "Device Protection",
                "Tech Support",
                "Streaming TV",
                "Streaming Movies",
                "Contract",
                "Paperless Billing",
                "Payment Method"
            ]

            num_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num",
                        num_pipeline,
                        numerical_columns
                    ),
                    (
                        "cat",
                        cat_pipeline,
                        categorical_columns
                    )
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(
        self
    ):
        try:

            df = pd.read_csv(
                self.config.interim_data_path
            )

            logging.info(
                "Splitting data into train and test sets"
            )

            target_column = "Churn Value"

            X = df.drop(
            columns=[target_column]
            )
            y = df[target_column]

            X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
            )

            logging.info(
                "Obtaining preprocessor object"
            )

            preprocessor = self.get_preprocessor()

            logging.info(
                "Transforming train and test sets"
            )

            X_train_processed = (
                preprocessor.fit_transform(
                    X_train
                )
            )

            X_test_processed = (
                preprocessor.transform(
                    X_test
                )
            )

            logging.info(
                "Saving preprocessor object"
            )

            save_object(
                file_path=self.config.preprocessor_path,
                obj=preprocessor
            )

            logging.info(
                "Data transformation completed successfully"
            )

            X_train_processed = pd.DataFrame(
                X_train_processed
            )

            X_test_processed = pd.DataFrame(
                X_test_processed
            )

            X_train_processed["Churn Value"] = (
                y_train.reset_index(drop=True)
            )

            X_test_processed["Churn Value"] = (
                y_test.reset_index(drop=True)
            )

            os.makedirs(
            os.path.dirname(
            self.config.train_processed_path
            ),
            exist_ok=True
            )

            X_train_processed.to_csv(
                self.config.train_processed_path,
                index=False
            )

            X_test_processed.to_csv(
                self.config.test_processed_path,
                index=False
            )

            return (
                self.config.train_processed_path,
                self.config.test_processed_path,
                self.config.preprocessor_path
            )
        except Exception as e:
            raise CustomException(e, sys)