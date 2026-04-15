from preprocessing import load_and_clean_data
from train_model import train_model
from evaluate import evaluate_model

# Load and clean data
df = load_and_clean_data('traffic_accidents.csv')

# Train model
model, X_test, y_test = train_model(df)

# Evaluate model
evaluate_model(model, X_test, y_test)