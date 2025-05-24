# app.py
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os
import numpy as np

from flask_cors import CORS  # Add this import at the top

app = Flask(__name__)
CORS(app)  # Add this line after creating the Flask app

# Rest of your code remains the same...

# Configuration
CSV_FILE = 'afl_data.csv'
MODEL_FILE = 'afl_goal_predictor.pkl'
TARGET_COLUMN = 'goals'

# Features to use for prediction (align with your frontend fields)
FEATURE_COLUMNS = [
    'kicks', 'marks', 'hand_balls', 'disposals', 'behinds',
    'hitouts', 'tackles', 'RB', 'IF', 'CL', 'CG', 'FF', 'FA',
    'BR', 'CP', 'UP', 'CM', 'MI', '1%', 'BO', 'GA'
]

# Field mapping between frontend and CSV columns
FIELD_MAPPING = {
    'KI': 'kicks',
    'MK': 'marks',
    'HB': 'hand_balls',
    'DI': 'disposals',
    'BH': 'behinds',
    'HO': 'hitouts',
    'TK': 'tackles',
    'RB': 'RB',
    'IF': 'IF',
    'CL': 'CL',
    'CG': 'CG',
    'FF': 'FF',
    'FA': 'FA',
    'BR': 'BR',
    'CP': 'CP',
    'UP': 'UP',
    'CM': 'CM',
    'MI': 'MI',
    '1%': '1%',
    'BO': 'BO',
    'GA': 'GA'
}

def clean_numeric(value):
    """Clean numeric values by removing newlines and converting to float"""
    if isinstance(value, str):
        value = value.replace('\n', '').strip()
        if value == '':  # Handle empty strings
            return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def load_and_prepare_data():
    """Load data from CSV and prepare for training"""
    try:
        df = pd.read_csv(CSV_FILE)
        
        # Ensure all required columns exist
        missing_cols = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] 
                       if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in CSV: {missing_cols}")
        
        # Clean all numeric columns
        for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
            df[col] = df[col].apply(clean_numeric)
        
        return df
    
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {str(e)}")

def train_model():
    """Train and save the prediction model"""
    df = load_and_prepare_data()
    
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model trained with MAE: {mae:.2f} goals")
    print(f"Average goals in dataset: {y.mean():.2f}")
    
    # Save model
    joblib.dump(model, MODEL_FILE)
    return model

# Load or train model
if not os.path.exists(MODEL_FILE):
    print("Training new model...")
    model = train_model()
else:
    print("Loading existing model...")
    model = joblib.load(MODEL_FILE)


# Load and clean test data
test_df = pd.read_csv("test.csv")

# Extract goal labels if available
goal_labels = test_df["goals"] if "goals" in test_df.columns else None
test_df = test_df.drop(columns=["player", "team", "Year", "goals", "games"], errors="ignore")

# Clean string values and convert to numeric
test_df = test_df.applymap(lambda x: str(x).strip().replace("\n", "") if isinstance(x, str) else x)
test_df = test_df.apply(pd.to_numeric, errors="coerce")
test_df = test_df.dropna()

# Predict for each row
predictions = model.predict(test_df)

if goal_labels is not None:
    for actual, predicted in zip(goal_labels, predictions):
        margin = abs(predicted - actual)
        percent_error = margin / actual * 100
        print(f"Actual: {actual}, Predicted: {predicted:.2f}, Margin: {margin:.2f}, % Error: {percent_error:.2f}%")
else:
    print("No actual goal labels found in test data.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from frontend
        data = request.get_json()
        
        # Prepare input data using field mapping
        input_data = {}
        for frontend_name, csv_name in FIELD_MAPPING.items():
            input_data[csv_name] = clean_numeric(data.get(frontend_name, 0))
        
        # Convert to DataFrame for prediction
        input_df = pd.DataFrame([input_data])[FEATURE_COLUMNS]
        
        # Make prediction
        prediction = model.predict(input_df)
        
        # Calculate confidence based on how far from mean we are
        df = load_and_prepare_data()
        mean_goals = df[TARGET_COLUMN].mean()
        std_goals = df[TARGET_COLUMN].std()
        
        # Confidence decreases as prediction gets further from mean
        z_score = abs(prediction[0] - mean_goals) / std_goals
        confidence = max(50, min(95, 90 - (z_score * 10)))
        
        return jsonify({
            'prediction': round(prediction[0], 1),
            'confidence': round(confidence, 1),
            'mean_goals': round(mean_goals, 1),
            'stats_entered': input_data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/retrain', methods=['POST'])
def retrain():
    """Endpoint to retrain the model with updated data"""
    try:
        global model
        model = train_model()
        return jsonify({'message': 'Model retrained successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)