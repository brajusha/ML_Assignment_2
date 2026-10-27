import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Machine Learning Classification Evaluator")
st.write("BITS Pilani WILP - M.Tech (AIML) - Assignment 2")

# a. Dataset upload option
uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"])

# b. Model selection dropdown
model_options = ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'Support Vector Machine']
selected_model = st.selectbox("Select Machine Learning Model", model_options)

if uploaded_file is not None:
    # Read the data
    df = pd.read_csv(uploaded_file)
    st.write("Test Data Preview:", df.head())
    
    if 'target' not in df.columns:
        st.error("The uploaded CSV must contain a 'target' column.")
    else:
        X_test = df.drop(columns=['target'])
        y_test = df['target']
        
        # Load the selected model
        model_filename = f"model/{selected_model.replace(' ', '_')}.pkl"
        try:
            model = joblib.load(model_filename)
            
            # Make Predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            # c. Calculate evaluation metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"
            
            # Display metrics
            st.subheader(f"Evaluation Metrics: {selected_model}")
            metrics_df = pd.DataFrame({
                "Metric": ["Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC Score"],
                "Value": [f"{acc:.4f}", f"{auc:.4f}" if auc != "N/A" else auc, f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", f"{mcc:.4f}"]
            })
            st.table(metrics_df)
            
            # d. Confusion matrix & Classification report
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            st.pyplot(fig)
            
            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred))
            
        except FileNotFoundError:
            st.error(f"Model file for {selected_model} not found. Please ensure the model is trained and saved in the 'model/' directory.")