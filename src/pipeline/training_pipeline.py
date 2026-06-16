import os
from src.components.data_transformation_f import (
    DataTransformation
)
print(os.getcwd())

transformer = DataTransformation()

train_path, test_path, preprocessor_path = (
    transformer.initiate_data_transformation()
)