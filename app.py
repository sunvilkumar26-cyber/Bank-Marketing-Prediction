import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="centered"
)

# Load the trained model and feature columns
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "bank_marketing_random_forest.pkl")
model_columns = joblib.load(BASE_DIR / "model_columns.pkl")

# Application title
st.title("🏦 Bank Marketing Prediction")
st.write(
    "Predict whether a customer is likely to subscribe "
    "to a bank term deposit."
)

st.info(
    "Enter customer details below. The model will predict "
    "the result using your trained Random Forest model."
)

# Load the original dataset to provide valid dropdown options
data_path = BASE_DIR / "bank-additional-full.csv"
bank_df = pd.read_csv(data_path, sep=";")

# Match the features used when training the model
excluded_columns = {
    "customer subscribed",
    "y",
    "duration",
    "age_group"
}

feature_columns = [
    col for col in bank_df.columns
    if col not in excluded_columns
]

# Create the input form
with st.form("customer_form"):

    st.subheader("Customer Details")

    customer_data = {}

    for column in feature_columns:

        if column not in bank_df.columns:
            continue

        if pd.api.types.is_numeric_dtype(bank_df[column]):

            customer_data[column] = st.number_input(
                column.replace("_", " ").title(),
                value=float(bank_df[column].median())
            )

        else:

            options = sorted(
                bank_df[column].dropna().astype(str).unique().tolist()
            )

            customer_data[column] = st.selectbox(
                column.replace("_", " ").title(),
                options=options
            )

    submitted = st.form_submit_button("Predict Subscription")

# Make prediction
if submitted:

    input_df = pd.DataFrame([customer_data])

    # Convert categorical values into numeric columns
    input_encoded = pd.get_dummies(input_df, dtype=int)

    # Match the columns and order used during model training
    input_encoded = input_encoded.reindex(
        columns=model_columns,
        fill_value=0
    )

    # Predict using the trained model
    prediction = model.predict(input_encoded)[0]

    # Display the result
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("The model predicts: Customer is likely to subscribe (YES).")
    else:
        st.warning("The model predicts: Customer is unlikely to subscribe (NO).")

    # Display probabilities when supported by the model
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_encoded)[0]

        yes_probability = probabilities[
            list(model.classes_).index(1)
        ]

        st.write(
            f"Estimated subscription probability: "
            f"**{yes_probability:.2%}**"
        )

st.caption(
    "This is a machine learning prediction, not a guarantee "
    "of customer behavior."
)