# Loading dataset
# Train-test split
# Saving train and test data
import pandas as pd
import os
from pathlib import Path

print(os.getcwd())

path = Path('dataset')/'raw'/'Telco_customer_churn.xlsx'

root_dir = Path(__file__).resolve().parents[2]

data_path = root_dir/path

def load_dataset():
    df = pd.read_excel(data_path)
    return df