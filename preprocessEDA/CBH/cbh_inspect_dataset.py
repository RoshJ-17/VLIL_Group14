
import pandas as pd
from pathlib import Path

file_path = Path("finalIntegratedDataset.csv")

# Read only a sample to inspect the structure
df = pd.read_csv(file_path, nrows=10000)

print("\n========== DATASET INFORMATION ==========")
print("Shape (sample):", df.shape)

print("\n========== COLUMN NAMES ==========")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== NUMERICAL STATISTICS ==========")
print(df.describe().T)

print("\n========== CATEGORICAL COLUMNS ==========")
print(df.select_dtypes(include=["object", "category"]).nunique())
