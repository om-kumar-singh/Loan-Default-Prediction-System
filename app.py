from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import json
from io import StringIO

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)

best_model = model_data['best_model']
scaler = model_data['scaler']
label_encoders = model_data['label_encoders']
feature_names = model_data['feature_names']
model_scores = model_data['model_scores']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        input_data = {
            'age': float(data['age']),
            'income': float(data['income']),
            'credit_score': float(data['credit_score']),
            'loan_amount': float(data['loan_amount']),
            'employment_type': data['employment_type'],
            'marital_status': data['marital_status'],
            'education': data['education']
        }

        df = pd.DataFrame([input_data])

        for col in ['employment_type', 'marital_status', 'education']:
            if col in label_encoders:
                df[col] = label_encoders[col].transform(df[col])

        df = df[feature_names]

        df_scaled = scaler.transform(df)

        prediction = best_model.predict(df_scaled)[0]
        prediction_proba = best_model.predict_proba(df_scaled)[0]

        confidence = float(max(prediction_proba)) * 100

        result = {
            'prediction': 'Default' if prediction == 1 else 'No Default',
            'confidence': round(confidence, 2),
            'risk_level': 'High' if prediction == 1 else 'Low'
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        content = file.read().decode('utf-8')
        df = pd.read_csv(StringIO(content))

        required_cols = ['age', 'income', 'credit_score', 'loan_amount',
                        'employment_type', 'marital_status', 'education']

        if not all(col in df.columns for col in required_cols):
            return jsonify({'error': 'Missing required columns'}), 400

        df_processed = df.copy()

        for col in ['employment_type', 'marital_status', 'education']:
            if col in label_encoders:
                df_processed[col] = label_encoders[col].transform(df_processed[col])

        df_processed = df_processed[feature_names]

        df_scaled = scaler.transform(df_processed)

        predictions = best_model.predict(df_scaled)
        prediction_probas = best_model.predict_proba(df_scaled)

        df['prediction'] = ['Default' if p == 1 else 'No Default' for p in predictions]
        df['confidence'] = [round(max(proba) * 100, 2) for proba in prediction_probas]
        df['risk_level'] = ['High' if p == 1 else 'Low' for p in predictions]

        results = df.to_dict('records')

        return jsonify({
            'results': results,
            'total': len(results),
            'defaults': int(sum(predictions)),
            'no_defaults': int(len(predictions) - sum(predictions))
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model_comparison')
def model_comparison():
    return jsonify(model_scores)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
