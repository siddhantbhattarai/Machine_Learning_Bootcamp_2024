from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

def load_model():
    try:
        model_path = os.path.join('models', 'random_forest_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        # Load the model data dictionary
        model_data = joblib.load(model_path)
        return model_data
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def preprocess_input(data, model_data):
    # Create DataFrame with single row
    df = pd.DataFrame([data])
    
    # Initialize a DataFrame with all possible columns from training
    full_df = pd.DataFrame(columns=model_data['feature_names'])
    
    # Update with actual values
    for col in df.columns:
        if col in full_df.columns:
            full_df[col] = df[col]
    
    # Fill missing columns with 0 (for one-hot encoded columns)
    full_df = full_df.fillna(0)
    
    # Scale the features using the saved scaler
    scaled_features = model_data['scaler'].transform(full_df)
    
    return scaled_features

# Load the model at startup
model_data = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model_data is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded properly. Please check server logs.'
        })
    
    try:
        # Get form data
        data = {
            'age': float(request.form['age']),
            'hypertension': int(request.form['hypertension']),
            'heart_disease': int(request.form['heart_disease']),
            'bmi': float(request.form['bmi']),
            'HbA1c_level': float(request.form['HbA1c_level']),
            'blood_glucose_level': float(request.form['blood_glucose_level']),
            f"gender_{request.form['gender']}": 1,
            f"smoking_history_{request.form['smoking_history']}": 1
        }
        
        # Preprocess input data
        input_data = preprocess_input(data, model_data)
        
        # Make prediction using the model from model_data
        prediction = model_data['model'].predict(input_data)[0]
        probability = model_data['model'].predict_proba(input_data)[0][1]
        
        result = {
            'prediction': int(prediction),
            'probability': round(float(probability) * 100, 2),
            'status': 'success'
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Error during prediction: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True)
