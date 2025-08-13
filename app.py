import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

model_path = "model.pkl"
scaler_path = "scaler.pkl"
data_path = "WineQT.csv"

# Load model and scaler
if os.path.exists(model_path) and os.path.exists(scaler_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
else:
    st.error("Model or scaler file not found. Please ensure 'model.pkl' and 'scaler.pkl' are in the same directory as this script.")
    st.stop()

# Load data
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    if "Id" in df.columns:
        df.drop("Id", axis=1, inplace=True)
else:
    st.error("WineQT.csv not found in the directory.")
    st.stop()

def predict_quality(features):
    # Use DataFrame for feature names
    feature_names = [col for col in df.columns if col != "quality"]
    features_df = pd.DataFrame([features], columns=feature_names)
    scaled = scaler.transform(features_df)
    prediction = model.predict(scaled)
    return "Good" if prediction[0] == 1 else "Bad"

st.sidebar.title("Wine Quality App")
option = st.sidebar.radio(
    "Select Page",
    ["Data Exploration", "Visualization", "Prediction", "Model Performance"]
)

if option == "Data Exploration":
    st.header("Data Exploration")
    st.write("First 5 rows of the dataset:")
    st.dataframe(df.head())
    st.write("Summary statistics:")
    st.dataframe(df.describe())
    st.write("Missing values:")
    st.dataframe(df.isnull().sum())

elif option == "Visualization":
    st.header("Visualization")
    st.subheader("Quality Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["quality"], bins=10, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

elif option == "Prediction":
    st.header("Wine Quality Prediction")
    st.write("Enter wine features below:")

    feature_names = [col for col in df.columns if col != "quality"]
    user_input = []
    for feature in feature_names:
        val = st.number_input(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
        user_input.append(val)

    if st.button("Predict Quality"):
        result = predict_quality(user_input)
        st.success(f"Predicted Wine Quality: {result}")

elif option == "Model Performance":
    st.header("Model Performance")
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    X = df.drop("quality", axis=1)
    y = df["quality"]
    # For demonstration, use train_test_split again
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test)
    preds = model.predict(X_test_scaled)

    st.write("Accuracy:", accuracy_score(y_test, preds))
    st.write("Classification Report:")
    st.text(classification_report(y_test, preds, zero_division=0))

    st.subheader("Confusion Matrix")
    fig3, ax3 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d", ax=ax3)
    st.pyplot(fig3)