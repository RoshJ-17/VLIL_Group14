import pandas as pd

FILE = "finalIntegratedDataset.csv"

df = pd.read_csv(
    FILE,
    usecols=["timestamp"],
    nrows=20,
    dtype={"timestamp": "string"}
)

print("First 20 raw timestamp values:")
for value in df["timestamp"]:
    print(repr(value))

print("\nTimestamp column information:")
print(df["timestamp"].dtype)
print(df["timestamp"].describe())
