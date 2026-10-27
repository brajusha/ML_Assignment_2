import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import joblib
import os

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# 1. Load Dataset (Breast Cancer: 569 instances, 30 features)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# 2. Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Save test data for Streamlit upload
test_data = X_test.copy()
test_data['target'] = y_test
test_data.to_csv('test_data.csv', index=False)
print("Saved test_data.csv successfully.")

# 4. Initialize the 5 Classification Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(probability=True, random_state=42) # probability=True is needed for AUC
}

# 5. Train and save models
for name, model in models.items():
    model.fit(X_train, y_train)
    # Save as .pkl file
    filename = f"model/{name.replace(' ', '_')}.pkl"
    joblib.dump(model, filename)
    print(f"Trained and saved: {name}")