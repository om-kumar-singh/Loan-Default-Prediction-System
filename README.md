# Loan Default Prediction System

A complete machine learning web application for predicting loan default risk using multiple ML algorithms.

## Features

- **Single Prediction**: Predict default risk for individual loan applications
- **Batch Prediction**: Upload CSV files for bulk predictions
- **Model Comparison**: View performance metrics of all trained models
- **Dashboard**: Exploratory Data Analysis visualizations
- **Multiple ML Models**: Decision Tree, Random Forest, Logistic Regression, KNN, Naive Bayes, and XGBoost
- **Automatic Model Selection**: System uses the best-performing model
- **Data Preprocessing**: Includes SMOTE for class balancing, label encoding, and feature scaling
- **Responsive UI**: Clean and modern interface that works on all devices

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. Clone or download this project

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Train the machine learning models:
```bash
python train_model.py
```

This will:
- Generate a synthetic dataset (5000 samples)
- Train 6 different ML models
- Apply SMOTE for class balancing
- Save the best model as `model.pkl`
- Create `loan_data.csv` with training data

4. Run the Flask application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

### Single Prediction

1. Navigate to the "Single Prediction" tab
2. Fill in customer details:
   - Age (18-100)
   - Annual Income
   - Credit Score (300-850)
   - Loan Amount
   - Employment Type
   - Marital Status
   - Education Level
3. Click "Predict Risk"
4. View the prediction result with confidence score

### Batch Prediction

1. Navigate to the "Batch Prediction" tab
2. Prepare a CSV file with the following columns:
   - age
   - income
   - credit_score
   - loan_amount
   - employment_type (Salaried/Self-Employed/Unemployed)
   - marital_status (Single/Married/Divorced)
   - education (High School/Bachelor/Master/PhD)
3. Upload the CSV file
4. Click "Process Batch"
5. View results in a table with statistics

### Model Comparison

1. Navigate to the "Model Comparison" tab
2. View training and testing accuracies for all models
3. See which model was selected as the best performer

### Dashboard

1. Click "Dashboard" in the navigation
2. View various EDA visualizations:
   - Credit Score Distribution
   - Default Rate by Employment Type
   - Loan Amount vs Income
   - Age Distribution
   - Feature Correlation Heatmap
   - Education Impact on Default Rates

## Models Used

The system trains and compares the following models:

1. **Decision Tree Classifier**
2. **Random Forest Classifier**
3. **Logistic Regression**
4. **K-Nearest Neighbors (KNN)**
5. **Naive Bayes**
6. **XGBoost Classifier**

The model with the highest test accuracy is automatically selected for predictions.

## Data Preprocessing

The application includes comprehensive preprocessing:

- **Label Encoding**: Converts categorical variables to numerical
- **SMOTE**: Balances class distribution in training data
- **Standard Scaling**: Normalizes feature values
- **Missing Value Handling**: Ensures data quality

## Project Structure

```
loan-default-prediction/
├── app.py                 # Flask application
├── train_model.py         # ML training pipeline
├── model.pkl             # Trained model (generated)
├── loan_data.csv         # Training dataset (generated)
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── templates/
│   ├── index.html       # Main prediction page
│   └── dashboard.html   # EDA dashboard
└── static/
    ├── style.css        # Stylesheet
    └── script.js        # JavaScript
```

## API Endpoints

- `GET /` - Main prediction page
- `GET /dashboard` - EDA dashboard
- `POST /predict` - Single prediction endpoint
- `POST /predict_batch` - Batch prediction endpoint
- `GET /model_comparison` - Model comparison data

## Technical Details

### Input Features

- **age**: Customer age (numeric)
- **income**: Annual income in dollars (numeric)
- **credit_score**: Credit score 300-850 (numeric)
- **loan_amount**: Requested loan amount (numeric)
- **employment_type**: Employment status (categorical)
- **marital_status**: Marital status (categorical)
- **education**: Education level (categorical)

### Output

- **Prediction**: Default or No Default
- **Confidence**: Probability score (0-100%)
- **Risk Level**: High or Low

## Performance

The system typically achieves:
- Training Accuracy: 85-95%
- Testing Accuracy: 80-90%

Actual performance depends on the best model selected during training.

## Notes

- The synthetic dataset is generated based on realistic default patterns
- Default prediction is influenced by credit score, loan-to-income ratio, employment status, and other factors
- The model is retrained each time you run `train_model.py`
- All predictions are made using the best-performing model from training

## Troubleshooting

If you encounter issues:

1. **Import errors**: Ensure all packages are installed: `pip install -r requirements.txt`
2. **Model not found**: Run `python train_model.py` first
3. **Port already in use**: Change the port in `app.py`: `app.run(debug=True, port=5001)`

## Future Enhancements

- Add real-time data collection
- Implement model retraining capability
- Add more visualization options
- Export predictions to PDF/Excel
- Add user authentication
- Deploy to cloud platform

