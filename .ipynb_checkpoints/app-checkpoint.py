import streamlit as st
import pandas as pd
import shap
import numpy as np
import pickle
import openai
import matplotlib.pyplot as plt

# Set up OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai_key"]

# Load model
with open("credit_risk_model.pkl", "rb") as f:
    model = pickle.load(f)

# Create SHAP explainer dynamically (no explainer.pkl)
explainer = shap.Explainer(model)

# App title
st.title("📊 Credit Risk Predictor & Business Analysis Assistant")

# User input section
st.header("👤 Enter Applicant Information")
age = st.slider("Age", 18, 75, 30)
income = st.number_input("Monthly Income", value=3000)
loan_amount = st.number_input("Loan Amount", value=10000)
credit_score = st.slider("Credit Score", 300, 850, 600)
employment_years = st.slider("Years of Employment", 0, 40, 5)
num_dependents = st.slider("Number of Dependents", 0, 10, 1)

user_data = pd.DataFrame([{
    "age": age,
    "income": income,
    "loan_amount": loan_amount,
    "credit_score": credit_score,
    "employment_years": employment_years,
    "num_dependents": num_dependents
}])

# Predict and explain
if st.button("🚀 Predict Credit Risk"):
    prediction = model.predict(user_data)
    prediction_proba = model.predict_proba(user_data)[0][1]
    risk_label = "High Risk 🔴" if prediction[0] == 1 else "Low Risk 🟢"

    st.subheader("📈 Prediction Result")
    st.markdown(f"**Risk Level:** {risk_label}")
    st.markdown(f"**Probability of Default:** {prediction_proba:.2%}")

    # SHAP Explanation
    st.subheader("🧠 SHAP Explanation")
    shap_values = explainer(user_data)

    fig, ax = plt.subplots()
    shap.plots.bar(shap_values[0], show=False)
    st.pyplot(fig)

    # Feature importance
    top_indices = np.argsort(np.abs(shap_values.values[0]))[::-1][:5]
    top_features = [user_data.columns[i] for i in top_indices]

    # Ask question
    st.subheader("💬 Ask a Business Question About This Prediction")
    user_question = st.text_area("Example: Why is this applicant high risk? What actions can we take?")

    if user_question:
        with st.spinner("Generating detailed business explanation..."):
            prompt = f"""
You are a senior financial analyst. Based on this credit risk prediction and SHAP explanation, provide a clear and structured business explanation in four sections:

1. **Prediction Summary** – Briefly explain the predicted risk and probability.
2. **Key Drivers** – Describe the most influential features: {', '.join(top_features)}.
3. **Business Implications** – Explain what this risk level means for lending or credit policy.
4. **Recommendation** – Provide suggestions or actions the business should consider.

Use the following data:
- Prediction: {"High risk" if prediction[0] == 1 else "Low risk"}
- Probability of default: {prediction_proba:.2%}
- Question: {user_question}

Be clear, professional, and insightful.
"""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful and insightful financial analyst."},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response['choices'][0]['message']['content']

        st.subheader("📘 Business Explanation")
        st.markdown(answer)
