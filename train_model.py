import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

n_samples = 5000

data = {
    'age': np.random.randint(18, 70, n_samples),
    'income': np.random.randint(20000, 150000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'loan_amount': np.random.randint(5000, 500000, n_samples),
    'employment_type': np.random.choice(['Salaried', 'Self-Employed', 'Unemployed'], n_samples, p=[0.6, 0.3, 0.1]),
    'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples, p=[0.3, 0.5, 0.2]),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.2, 0.4, 0.3, 0.1])
}

df = pd.DataFrame(data)

def determine_default(row):
    score = 0

    if row['credit_score'] < 580:
        score += 40
    elif row['credit_score'] < 670:
        score += 25
    elif row['credit_score'] < 740:
        score += 10

    loan_to_income = row['loan_amount'] / row['income']
    if loan_to_income > 5:
        score += 30
    elif loan_to_income > 3:
        score += 15

    if row['employment_type'] == 'Unemployed':
        score += 35
    elif row['employment_type'] == 'Self-Employed':
        score += 10

    if row['age'] < 25:
        score += 10
    elif row['age'] > 60:
        score += 15

    if row['marital_status'] == 'Divorced':
        score += 5

    score += np.random.randint(-15, 15)

    return 1 if score > 50 else 0

df['default'] = df.apply(determine_default, axis=1)

print(f"Dataset created with {len(df)} samples")
print(f"Default distribution:\n{df['default'].value_counts()}")
print(f"Default rate: {df['default'].mean()*100:.2f}%")

label_encoders = {}
categorical_cols = ['employment_type', 'marital_status', 'education']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop('default', axis=1)
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE - Training set distribution:\n{pd.Series(y_train_balanced).value_counts()}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss')
}

model_scores = {}
trained_models = {}

print("\n" + "="*60)
print("TRAINING AND EVALUATING MODELS")
print("="*60)

for name, model in models.items():
    print(f"\n{name}:")
    model.fit(X_train_scaled, y_train_balanced)

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train_balanced, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    print(f"  Training Accuracy: {train_accuracy*100:.2f}%")
    print(f"  Testing Accuracy: {test_accuracy*100:.2f}%")

    model_scores[name] = {
        'train_accuracy': round(train_accuracy * 100, 2),
        'test_accuracy': round(test_accuracy * 100, 2)
    }

    trained_models[name] = model

best_model_name = max(model_scores, key=lambda x: model_scores[x]['test_accuracy'])
best_model = trained_models[best_model_name]

print("\n" + "="*60)
print(f"BEST MODEL: {best_model_name}")
print(f"Test Accuracy: {model_scores[best_model_name]['test_accuracy']:.2f}%")
print("="*60)

y_pred = best_model.predict(X_test_scaled)
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

model_data = {
    'best_model': best_model,
    'best_model_name': best_model_name,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'feature_names': list(X.columns),
    'model_scores': model_scores
}

with open('model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"\nModel saved as model.pkl")

df_original = pd.DataFrame(data)
df_original['default'] = df['default']
df_original.to_csv('loan_data.csv', index=False)
print("Dataset saved as loan_data.csv")
