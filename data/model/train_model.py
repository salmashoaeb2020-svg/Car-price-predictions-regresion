import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. Load Dataset
# ==========================================

data_path = "../data/cars.csv"

df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print(f"Number of cars: {len(df)}")


# ==========================================
# 2. Remove Missing Values
# ==========================================

df = df.dropna()

print(f"Dataset after cleaning: {len(df)}")


# ==========================================
# 3. Features and Target
# ==========================================

X = df[
    [
        "Brand",
        "Model",
        "Year",
        "Kilometers",
        "Condition",
        "Trim"
    ]
]

y = df["Price"]


# ==========================================
# 4. Categorical / Numerical Features
# ==========================================

categorical_features = [
    "Brand",
    "Model",
    "Condition",
    "Trim"
]

numerical_features = [
    "Year",
    "Kilometers"
]


# ==========================================
# 5. Preprocessing
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ==========================================
# 6. Regression Model
# ==========================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=None
)


# ==========================================
# 7. Create Pipeline
# ==========================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ==========================================
# 8. Train / Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 9. Train Model
# ==========================================

print("\nTraining model...")

pipeline.fit(X_train, y_train)

print("Model trained successfully!")


# ==========================================
# 10. Predictions
# ==========================================

y_pred = pipeline.predict(X_test)


# ==========================================
# 11. Evaluation
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n========== Model Evaluation ==========")

print(f"MAE  : {mae:,.2f} EGP")
print(f"RMSE : {rmse:,.2f} EGP")
print(f"R²   : {r2:.4f}")


# ==========================================
# 12. Save Model
# ==========================================

model_path = "car_price_model.pkl"

joblib.dump(pipeline, model_path)

print("\nModel saved successfully!")
print(f"Saved as: {model_path}")
