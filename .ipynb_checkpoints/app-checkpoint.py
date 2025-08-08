import streamlit as st
import pandas as pd
import shap
import numpy as np
import pickle
import openai
import matplotlib.pyplot as plt

# Set up OpenAI API key
openai.api_key = st.secrets["openai_key"]

# Load trained model and explainer
with open("credit_risk_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("explainer.pkl", "rb") as f:
    explainer = pickle.load(f)

st.title("📊 Credit Risk Predictor & Business Explanation Tool")

# User input
st.header("Enter Applicant Information")
age = st.slider("Age", 18, 75, 30)
income = st.number_input("Monthly Income", value=3000)
loan_amount = st.number_input("Loan Amount", value=10000)
credit_score = st.slider("Credit Score", 300, 850, 600)
employment_years = st.slider("Years of Employment", 0, 40, 5)
num_dependents = st.slider("Number of Dependents", 0, 10, 1)

# Create dataframe from input
user_data = pd.DataFrame([{
    "age": age,
    "income": income,
    "loan_amount": loan_amount,
    "credit_score": credit_score,
    "employment_years": employment_years,
    "num_dependents": num_dependents
}])

# Predict & explain
if st.button("✨ Predict Credit Risk! ✨"):
    prediction = model.predict(user_data)
    prediction_proba = model.predict_proba(user_data)[0][1]

    risk_label = "High Risk 🔴" if prediction[0] == 1 else "Low Risk 🟢"
    st.subheader("📈 Prediction Result:")
    st.markdown(f"**Risk Assessment:** {risk_label}")
    st.markdown(f"**Probability of Default:** {prediction_proba:.2%}")

    # SHAP explanation
    shap_values = explainer.shap_values(user_data)
    st.subheader("🧠 Model Explanation (SHAP)")

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, user_data, plot_type="bar", show=False)
    st.pyplot(fig)

    # Top features
    top_indices = np.argsort(np.abs(shap_values[0]))[::-1][:5]
    top_features = [user_data.columns[i] for i in top_indices]

    # User question input
    user_question = st.text_area("💬 Ask a business question about this prediction:")

    if user_question:
        with st.spinner("Generating business explanation..."):
            prompt = f"""
You are a financial analyst. Based on the following prediction and SHAP explanation, answer the user's question in business terms:

- Prediction result: {"High risk" if prediction[0] == 1 else "Low risk"}
- Probability of default: {prediction_proba:.2%}
- Top influencing features: {', '.join(top_features)}
- Question: {user_question}

Answer clearly in 100-150 words to help business decision-makers understand the result.
"""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful financial analyst."},
                    {"role": "user", "content": prompt}
                ]
            )
            explanation = response['choices'][0]['message']['content']

        st.subheader("📘 Business Explanation")
        st.write(explanation)
