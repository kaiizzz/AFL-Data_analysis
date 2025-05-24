import pandas as pd
from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load and clean the data
def load_data():
    df = pd.read_csv("afl_data.csv")

    # Drop unneeded columns
    df = df.drop(columns=["player", "team", "Year"], errors="ignore")

    # Strip whitespace and newlines from all cells
    df = df.applymap(lambda x: str(x).strip().replace("\n", "") if isinstance(x, str) else x)

    # Convert all columns to numeric, force errors to NaN
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop rows with any NaNs (caused by bad values)
    df = df.dropna()

    return df
# Train model
df = load_data()
print(df.head())  # Debugging line to check the data
X = df.drop(columns=["goals"])
y = df["goals"]
model = LinearRegression()
model.fit(X, y)

# # Load and clean test data
# test_df = pd.read_csv("test.csv")

# # Extract goal labels if available
# goal_labels = test_df["goals"] if "goals" in test_df.columns else None
# test_df = test_df.drop(columns=["player", "team", "Year", "goals"], errors="ignore")

# # Clean string values and convert to numeric
# test_df = test_df.applymap(lambda x: str(x).strip().replace("\n", "") if isinstance(x, str) else x)
# test_df = test_df.apply(pd.to_numeric, errors="coerce")
# test_df = test_df.dropna()

# # Predict for each row
# predictions = model.predict(test_df)

# # Print evaluation metrics
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# if goal_labels is not None:
#     for actual, predicted in zip(goal_labels, predictions):
#         margin = abs(predicted - actual)
#         percent_error = margin / actual * 100
#         print(f"Actual: {actual}, Predicted: {predicted:.2f}, Margin: {margin:.2f}, % Error: {percent_error:.2f}%")
# else:
#     print("No actual goal labels found in test data.")


# Define prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()
        input_df = pd.DataFrame([data])
        # Convert all to numeric
        input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        prediction = model.predict(input_df)[0]
        return jsonify({"predicted_goals": round(prediction, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
