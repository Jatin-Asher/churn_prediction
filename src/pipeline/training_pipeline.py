import os

from src.components.data_transformation_f import DataTransformation
from src.components.model_training import ModelTrainer

print(os.getcwd())

transformer = DataTransformation()

train_path, test_path, preprocessor_path = (
    transformer.initiate_data_transformation()
)

trainer = ModelTrainer()

trainer.initiate_model_training(
    train_path=train_path,
    test_path=test_path
)