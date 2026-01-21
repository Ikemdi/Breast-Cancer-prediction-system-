import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model():
    # 1. Load Dataset (Ensure breast_cancer_data.csv is in the root or same folder)
    # Using the standard UCI Breast Cancer Wisconsin (Diagnostic) headers
    try:
        df = pd.read_csv('breast_cancer_data.csv')
    except FileNotFoundError:
        print("Error: breast_cancer_data.csv not found. Please ensure the dataset is in the directory.")
        return

    # 2. Feature Selection (Strictly 5 features as per instructions)
    selected_features = [
        'radius_mean', 
        'texture_mean', 
        'perimeter_mean', 
        'area_mean', 
        'smoothness_mean'
    ]
    
    X = df[selected_features]
    y = df['diagnosis']

    # 3. Preprocessing
    # Handling missing values (if any)
    X = X.fillna(X.mean())

    # Encoding target variable (Malignant/Benign to 1/0)
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Feature Scaling (Mandatory for Logistic Regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 5. Implementation (Logistic Regression)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 6. Evaluation
    y_pred = model.predict(X_test)
    print("--- Model Evaluation ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")

    # 7. Save the model and scaler
    # We save the scaler because the web app must scale user input the same way
    joblib.dump(model, 'model/breast_cancer_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    print("\nModel and Scaler saved successfully in /model/ folder.")

if __name__ == "__main__":
    train_model()