
import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"

#Load the raw dataset

df = pd.read_csv(RAW_PATH)

# Validate the expected columns are present or not

expected_columns = ['CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier', 
                    'Occupation', 'Gender', 'NumberOfPersonVisiting', 'PreferredProperty,
]

missing = [c for c in expected_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {', '.join(missing)}")

print("Dataset registered successfully")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"Columns:", list(df.columns))
