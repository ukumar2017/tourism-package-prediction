
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the csv file
df = pd.read_csv("tourism_project/data/tourism.csv")

# Check unique values
print(df['Geneder'].unique())

# Correct inconsistent values
df['Gender'] = df['Gender'].replace({
    'Fe Male': 'Female'
})

df.drop(columns=['CustomerID'], inplace=True)

X = df.drop(columns=['ProdTaken'])
y = df['ProdTaken']

# Split the data into 80% training and 20% testing sets
#stratify=y keeps the (imbalanced) failure ratio consistent across splits to ensurs that both
# training and testing sets have the same proportion of buyers versus non-buyers
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Save the split datasets locally for subsequent pipeline steps scaling
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Data prepared: train/test splites written.")
