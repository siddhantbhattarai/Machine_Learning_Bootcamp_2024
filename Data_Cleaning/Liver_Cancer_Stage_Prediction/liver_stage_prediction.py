from flask import Flask, request, jsonify, render_template
import os
import pickle
import numpy as np
from joblib import load

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model_path = "models/best_model.pkl"
if os.path.exists(model_path):
    try:
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
    except pickle.UnpicklingError:
        model = load(model_path)  # Try loading with joblib
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Information about each stage
stage_info = {
    0: "Stage 0: Early stage with no significant spread. Regular monitoring and lifestyle adjustments are recommended.",
    1: "Stage 1: The cancer is localized. Early treatment options such as surgery or ablation may be effective.",
    2: "Stage 2: The cancer has started to grow or spread slightly. Treatment may include surgery and targeted therapies.",
    3: "Stage 3: Advanced stage with regional spread. A combination of therapies such as surgery, chemotherapy, or immunotherapy may be necessary.",
    4: "Stage 4: Cancer has metastasized to other parts of the body. Treatment focuses on managing symptoms and prolonging quality of life."
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Convert all inputs to float
        features = np.array([
            float(data.get("age", 0)),
            float(data.get("gender", 0)),
            float(data.get("alt", 0)),
            float(data.get("ast", 0)),
            float(data.get("total_bilirubin", 0)),
            float(data.get("direct_bilirubin", 0)),
            float(data.get("albumin", 0)),
            float(data.get("alcohol", 0)),
            float(data.get("hepatitis", 0)),
            float(data.get("cirrhosis", 0))
        ]).reshape(1, -1)

        # Add debugging print statements
        print("Input features:", features)
        
        # Make prediction
        prediction = model.predict(features)
        predicted_stage = int(prediction[0])
        
        print("Raw prediction:", prediction)
        print("Predicted stage:", predicted_stage)

        # Validate predicted stage
        if predicted_stage not in stage_info:
            return jsonify({
                "error": f"Invalid prediction: {predicted_stage}",
                "status": "error"
            }), 400

        # Get stage information
        info = stage_info[predicted_stage]

        return jsonify({
            "stage": predicted_stage,
            "info": info,
            "status": "success"
        })

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

if __name__ == "__main__":
    app.run(debug=True)