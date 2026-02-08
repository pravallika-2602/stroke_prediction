import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("stroke_data.csv")

# Convert categorical to numbers (must match FastAPI)
df["gender"] = df["gender"].map({"Male":1, "Female":0})
df["ever_married"] = df["ever_married"].map({"Yes":1, "No":0})
df["work_type"] = df["work_type"].map({
    "Private":0, "Self-employed":1, "Govt_job":2, "children":3
})
df["Residence_type"] = df["Residence_type"].map({"Urban":1, "Rural":0})
df["smoking_status"] = df["smoking_status"].map({
    "never smoked":0, "formerly smoked":1, "smokes":2, "Unknown":3
})

X = df.drop("stroke", axis=1)
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "stroke_xgb.pkl")
print("✅ New correct model saved!")
