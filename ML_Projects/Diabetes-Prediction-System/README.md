# Diabetes Prediction System

## Overview

The **Diabetes Prediction System** is a web-based application designed to predict the risk of diabetes in individuals based on various health indicators. It utilizes a machine learning model (Random Forest) trained on relevant health data to provide real-time predictions.

## Features

- **User-Friendly Interface**: An intuitive form for inputting health data.
- **Real-Time Predictions**: Instant feedback using a machine learning model.
- **Responsive Design**: Accessible on desktop and mobile devices.
- **Visualization**: Easy-to-interpret results showing the probability of diabetes.

## Tech Stack

- **Frontend**: HTML, CSS (Bootstrap), JavaScript (jQuery)
- **Backend**: Flask (Python)
- **Machine Learning**: Random Forest Classifier (Scikit-learn)
- **Model Serialization**: Pickle

## Directory Structure

```
Diabetes-Prediction-System/
├── Notebook/
│   └── Diabetes-Prediction-System.ipynb   # Jupyter Notebook for model training
├── models/
│   └── random_forest_model.pkl            # Trained Random Forest model file
├── templates/
│   └── index.html                         # HTML file for the web interface
├── test cases/
│   ├── negative cases.png                 # Example of negative prediction
│   └── positive cases.png                 # Example of positive prediction
└── diabetes_pred.py                       # Flask application file
```

## Installation

### Prerequisites

- Python 3.x
- Flask
- Scikit-learn
- Pandas
- Numpy

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Diabetes-Prediction-System.git
   cd Diabetes-Prediction-System
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Create a `requirements.txt` file with the following content if it's not already present:
   ```
   Flask
   scikit-learn
   pandas
   numpy
   ```

4. **Run the Flask application**:
   ```bash
   python diabetes_pred.py
   ```

5. **Open the application**:
   Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

## Usage

1. Enter the relevant health data in the form:
   - Gender
   - Age
   - Hypertension (Yes/No)
   - Heart Disease (Yes/No)
   - Smoking History
   - BMI
   - HbA1c Level
   - Blood Glucose Level

2. Click the **Predict** button to get the results.
3. The system will display whether the user has a potential risk of diabetes along with the prediction probability.

## Model Training

The `Diabetes-Prediction-System.ipynb` notebook contains the code for:
- Data exploration and preprocessing
- Training the Random Forest model
- Evaluation metrics like accuracy, precision, recall, etc.
- Saving the trained model as `random_forest_model.pkl`

## Examples

### Negative Case
![Negative Case](test%20cases/negative%20cases.png)

### Positive Case
![Positive Case](test%20cases/positive%20cases.png)

## Future Improvements

- Adding more features like physical activity, diet, and family history.
- Implementing additional machine learning models for improved accuracy.
- Enhancing the UI for a better user experience.
- Integrating data visualization for better insights.
