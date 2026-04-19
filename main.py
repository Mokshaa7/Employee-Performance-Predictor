from src.data_generation import generate_data
from src.preprocessing import preprocess_data
from src.model import train_model
from src.evaluate import evaluate_model
from src.utils import plot_feature_importance

import joblib
import os

# Create folders if not exist
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Step 1: Generate Data
generate_data()

# Step 2: Preprocess
X, y = preprocess_data()

# Step 3: Train Model
model, X_test, y_test = train_model(X, y)

# Step 4: Evaluate
accuracy = evaluate_model(model, X_test, y_test)

# Step 5: Feature Importance
plot_feature_importance(model, X)

# Step 6: Save Model
joblib.dump(model, "models/model.pkl")

print("Model saved successfully!") 