import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Sample student data
data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 6],
    "attendance":  [40, 50, 60, 65, 75, 80, 90, 95, 45, 85],
    "internal":    [10, 15, 20, 25, 30, 35, 40, 45, 12, 38],
    "result":      [0, 0, 0, 0, 1, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[["study_hours", "attendance", "internal"]]
y = df["result"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model trained and saved successfully")