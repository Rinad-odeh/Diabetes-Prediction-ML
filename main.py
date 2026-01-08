import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import os


df = pd.read_csv('diabetes.csv')

MODEL_EXISTS = os.path.exists('best_model.pkl') and os.path.exists('scaler.pkl')  

if not MODEL_EXISTS:
    print("Plotting feature distributions (Histograms)...")
    df.hist(bins=50, figsize=(12, 10))
    plt.suptitle("Feature Distributions (Before Cleaning)")
    plt.show()

    print("Plotting correlation heatmap...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)

medians = df.median()
df = df.fillna(medians)
joblib.dump(medians, 'medians.pkl')
   
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')

if not MODEL_EXISTS:
    print("\nTraining models and searching for the best parameters...")

    models_params = {
        'Logistic Regression': {
            'model': LogisticRegression(class_weight='balanced', random_state=42),
            'params': {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
        },
        'Random Forest': {
            'model': RandomForestClassifier(class_weight='balanced', random_state=42),
            'params': {'n_estimators': [50, 100], 'max_depth': [10, 20]}
        },
        'SVM': {
            'model': SVC(class_weight='balanced', probability=True, random_state=42),
            'params': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
        }
    }

    best_f1 = 0
    best_model_overall = None
    best_name = ""
    roc_data = {}

    for name, mp in models_params.items():
        clf = GridSearchCV(
            mp['model'],
            mp['params'],
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        clf.fit(X_train_scaled, y_train)

        cv_f1 = clf.best_score_
        y_pred = clf.best_estimator_.predict(X_test_scaled)
        test_f1 = f1_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        print(f" -> {name}: CV-F1={cv_f1:.2%} | Test-F1={test_f1:.2%} | Recall={rec:.2%}")

        if cv_f1 > best_f1:
            best_f1 = cv_f1
            best_model_overall = clf.best_estimator_
            best_name = name

        y_prob = (
            clf.best_estimator_.predict_proba(X_test_scaled)[:, 1]
            if hasattr(clf.best_estimator_, "predict_proba")
            else clf.best_estimator_.decision_function(X_test_scaled)
        )

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        roc_data[name] = (fpr, tpr, roc_auc)

    print(f"\nThe best model (CV-F1 Optimized) is: {best_name}")

 
    y_final_pred = best_model_overall.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_final_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Healthy', 'Diabetic'],
        yticklabels=['Healthy', 'Diabetic']
    )
    plt.title(f'Confusion Matrix for {best_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

  
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, roc_auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.show()

    joblib.dump(best_model_overall, 'best_model.pkl')
    print("\nModel and Scaler saved successfully.")


def predict_new_patient_quick(preg, gluc, bp, skin, ins, bmi, dpf, age):

    inputs = [preg, gluc, bp, skin, ins, bmi, dpf, age]
    if any(i < 0 for i in inputs):
        print(f"Error: Input {inputs} contains negative values.")
        return

    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    saved_medians = joblib.load('medians.pkl')

    columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]

    patient_data = pd.DataFrame(
        [[preg, gluc, bp, skin, ins, bmi, dpf, age]],
        columns=columns
    )

    medical_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in medical_cols:
        if patient_data[col][0] == 0:
            patient_data[col] = saved_medians[col]

    patient_scaled = scaler.transform(patient_data)
    prediction = model.predict(patient_scaled)
    prob = model.predict_proba(patient_scaled)[:, 1]

    result = "DIABETIC" if prediction[0] == 1 else "HEALTHY"
    print(f"The model predicts: {result} (probability = {prob[0]:.2f})")

print("\n--- Model Prediction Results ---")
predict_new_patient_quick(5, 160, 85, 30, 150, 35.2, 0.8, 45)
predict_new_patient_quick(1, 85, 66, 20, 90, 22.5, 0.25, 30)
predict_new_patient_quick(2, 120, 66, 20, 4, 35, 1, 33)
predict_new_patient_quick(2, 120, 67, -1, 8, 22.5, 0.8, 35)
