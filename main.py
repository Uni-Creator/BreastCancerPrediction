import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load trained models
model_files = {
    "RandomForest": "models/model_RandomForest.pkl",
    "SVM": "models/model_SVM.pkl",
    "LogisticRegression": "models/model_LogisticRegression.pkl",
    "KNN": "models/model_KNN.pkl",
    "DecisionTree": "models/model_DecisionTree.pkl",
    "NaiveBayes": "models/model_NaiveBayes.pkl"
}
models = {name: pickle.load(open(file, "rb")) for name, file in model_files.items()}

# Load dataset to get random samples
df = pd.read_csv("breast-cancer.csv")
X = df.iloc[:, 2:]  # Features only
y = df.iloc[:, 1]   # Diagnosis (M = Malignant, B = Benign)

# Streamlit UI
st.title("🔬 Breast Cancer Prediction App")
st.write("Enter the 30 feature values or use a random row from the dataset.")

# Feature names
feature_names = X.columns.tolist()

# Initialize session state
if "autofilled" not in st.session_state:
    st.session_state["autofilled"] = False
    st.session_state["original_diagnosis"] = "Unknown"
    for feature in feature_names:
        st.session_state[feature] = 0.0

# Autofill function
def autofill():
    random_index = np.random.randint(0, len(X))  # Pick a random row index
    random_row = X.iloc[random_index].values.flatten().astype(float)  # Convert row to float
    original_diagnosis = "Malignant" if y.iloc[random_index] == "M" else "Benign"

    for i, feature in enumerate(feature_names):
        st.session_state[feature] = random_row[i]
    st.session_state["original_diagnosis"] = original_diagnosis
    st.session_state["autofilled"] = True
    st.rerun()  # Refresh UI

# Autofill button
if st.button("🔄 Autofill Random Row"):
    autofill()

# Arrange inputs in a 3x10 grid
cols = st.columns(3)
user_inputs = {}

for i, feature in enumerate(feature_names):
    col_index = i % 3
    with cols[col_index]:
        user_inputs[feature] = st.number_input(
            feature, 
            value=float(st.session_state[feature]),  
            format="%.6f", 
            key=feature
        )

# Prediction button
if st.button("🔍 Predict"):
    input_values = np.array([user_inputs[feature] for feature in feature_names]).reshape(1, -1)
    
    predictions = {}
    confidence_scores = {}

    # Perform predictions using all models
    for model_name, model in models.items():
        if hasattr(model, "predict_proba"):  # Check if model supports predict_proba()
            pred_proba = model.predict_proba(input_values)[0]  # Get probability scores
            confidence = round(max(pred_proba) * 100, 2)  # Confidence as a percentage
            prediction = "Benign" if pred_proba[1] > 0.5 else "Malignant"
        else:
            prediction = model.predict(input_values)[0]  # Models like SVM might not support proba
            confidence = "N/A"

        predictions[model_name] = prediction
        confidence_scores[model_name] = confidence

    # Display predictions
    st.subheader("📊 Model Predictions")
    results_df = pd.DataFrame({
        "Model": predictions.keys(),
        "Prediction": predictions.values(),
        "Confidence (%)": confidence_scores.values()
    })
    st.dataframe(results_df)

    # Display original diagnosis if autofilled
    if st.session_state["autofilled"]:
        st.write(f"**Original Diagnosis:** {st.session_state['original_diagnosis']}")

    # Plot confidence scores as a bar chart with models on the x-axis
    fig, ax = plt.subplots(figsize=(10, 6))
    models_with_confidence = [m for m, c in confidence_scores.items() if c != "N/A"]
    confidence_values = [c for c in confidence_scores.values() if c != "N/A"]

    ax.bar(models_with_confidence, confidence_values, color="skyblue")
    ax.set_xlabel("Models")
    ax.set_ylabel("Confidence Score (%)")
    ax.set_title("Model Confidence Scores")
    plt.ylim(0, 100)  # Confidence scores range from 0 to 100
    plt.xticks(rotation=45)  # Rotate model names for better visibility
    st.pyplot(fig)
