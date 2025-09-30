import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
)

st.set_page_config(page_title="Healthcare Data Analytics", layout="wide")

st.title("🏥 Healthcare Data Analytics - Diabetes Prediction")
st.markdown("This app analyzes healthcare data, builds ML models, and predicts diabetes risk.")

@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

df = load_data()

# Sidebar
st.sidebar.header("User Controls")
show_eda = st.sidebar.checkbox("Show Exploratory Data Analysis", value=True)
show_models = st.sidebar.checkbox("Train & Evaluate Models", value=True)
show_predict = st.sidebar.checkbox("Try Patient Prediction", value=True)

# Show Dataset
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Exploratory Data Analysis (EDA)
if show_eda:
    st.subheader("🔎 Exploratory Data Analysis")

    st.write("**Distribution of Diabetes (Outcome)**")
    st.bar_chart(df["Outcome"].value_counts())

    st.write("**Correlation Heatmap**")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Train/Test Split
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Models & Evaluation
if show_models:
    st.subheader("🤖 Model Training & Evaluation")

    # Logistic Regression
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    y_pred_log = log_model.predict(X_test)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    col1, col2 = st.columns(2)
    with col1:
        st.write("🔹 **Logistic Regression Results**")
        st.write("Accuracy:", accuracy_score(y_test, y_pred_log))
        st.text(classification_report(y_test, y_pred_log))

    with col2:
        st.write("🔹 **Random Forest Results**")
        st.write("Accuracy:", accuracy_score(y_test, y_pred_rf))
        st.text(classification_report(y_test, y_pred_rf))

    # Confusion Matrix
    st.write("**Confusion Matrix (Random Forest)**")
    cm = confusion_matrix(y_test, y_pred_rf)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ROC Curve
    st.write("**ROC Curve (Random Forest)**")
    y_prob_rf = rf_model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob_rf):.2f}")
    ax.plot([0,1],[0,1],'--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Random Forest")
    ax.legend()
    st.pyplot(fig)

    # Feature Importance
    st.write("**Feature Importance (Random Forest)**")
    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values(by="Importance", ascending=False)
    st.bar_chart(feat_df.set_index("Feature"))

# Patient Prediction Form
if show_predict:
    st.subheader("🧑‍⚕️ Predict Patient Diabetes Risk")

    # Input Form
    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            Pregnancies = st.number_input("Pregnancies", 0, 20, 1)
            Glucose = st.number_input("Glucose", 0, 200, 100)
            BloodPressure = st.number_input("Blood Pressure", 0, 150, 70)
        with col2:
            SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)
            Insulin = st.number_input("Insulin", 0, 900, 80)
            BMI = st.number_input("BMI", 0.0, 70.0, 25.0)
        with col3:
            DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
            Age = st.number_input("Age", 1, 120, 30)

        submitted = st.form_submit_button("Predict")

    if submitted:
        patient_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                  Insulin, BMI, DiabetesPedigreeFunction, Age]])
        prediction = rf_model.predict(patient_data)[0]
        prob = rf_model.predict_proba(patient_data)[0][1]

        if prediction == 1:
            st.error(f"⚠️ High Risk of Diabetes (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Low Risk of Diabetes (Probability: {prob:.2f})")
