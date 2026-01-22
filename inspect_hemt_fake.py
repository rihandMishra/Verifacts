import pandas as pd

# CHANGE THIS PATH to where you extracted the dataset
DATA_PATH = "hemt_fake.csv"

df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nLabel distribution:\n", df["label"].value_counts())

print("\nSample rows:\n")
print(df.head(3))
