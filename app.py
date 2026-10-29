import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="centered"
)

# --------------------------------------------------
# 2. Load Saved Model and Feature Columns
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(
    BASE_DIR / "bank_marketing_random_forest.pkl"
)

model_columns = joblib.load(
    BASE_DIR / "model_columns.pkl"
)

# --------------------------------------------------
# 3. Application Title
# --------------------------------------------------

st.title("🏦 Bank Marketing Prediction")

st.write(
    "Predict whether a customer is likely to subscribe "
    "to a bank term deposit."
)

st.info(
    "Enter the customer details below and click "
    "'Predict Subscription' to get a prediction."
)

# --------------------------------------------------
# 4. Define Input Options
# --------------------------------------------------

categorical_options = {
    "job": [
        "admin.", "blue-collar", "entrepreneur", "housemaid",
        "management", "retired", "self-employed", "services",
        "student", "technician", "unemployed", "unknown"
    ],

    "marital": [
        "divorced", "married", "single", "unknown"
    ],

    "education": [
        "basic.4y", "basic.6y", "basic.9y",
        "high.school", "illiterate",
        "professional.course", "university.degree", "unknown"
    ],

    "default": [
        "no", "unknown", "yes"
    ],

    "housing": [
        "no", "unknown", "yes"
    ],

    "loan": [
        "no", "unknown", "yes"
    ],

    "contact": [
        "cellular", "telephone"
    ],

    "month": [
        "mar", "apr", "may", "jun", "jul", "aug",
        "sep", "oct", "nov", "dec"
    ],

    "day_of_week": [
        "mon", "tue", "wed", "thu", "fri"
    ],

    "poutcome": [
        "failure", "nonexistent", "success"
    ]
}

# Numeric input defaults
numeric_defaults = {
    "age": 38,
    "campaign": 2,
    "pdays": 999,
    "previous": 0,
    "emp.var.rate": 1.1,
    "cons.price.idx": 93.994,
    "cons.conf.idx": -41.8,
    "euribor3m": 4.857,
    "nr.employed": 5191.0
}

# --------------------------------------------------
# 5. Customer Input Form
# --------------------------------------------------

with st.form("customer_form"):

    st.subheader("Customer Details")

    customer_data = {}

    st.markdown("### Personal Information")

    customer_data["age"] = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=38
    )

    for column in ["job", "marital", "education"]:
        customer_data[column] = st.selectbox(
            column.replace("_", " ").title(),
            categorical_options[column]
        )

    st.markdown("### Existing Financial Products")

    for column in ["default", "housing", "loan"]:
        customer_data[column] = st.selectbox(
            column.replace("_", " ").title(),
            categorical_options[column]
        )

    st.markdown("### Campaign Information")

    customer_data["contact"] = st.selectbox(
        "Contact Type",
        categorical_options["contact"]
    )

    customer_data["month"] = st.selectbox(
        "Last Contact Month",
        categorical_options["month"]
    )

    customer_data["day_of_week"] = st.selectbox(
        "Last Contact Day",
        categorical_options["day_of_week"]
    )

    customer_data["campaign"] = st.number_input(
        "Contacts During Current Campaign",
        min_value=1,
        max_value=100,
        value=2
    )

    customer_data["pdays"] = st.number_input(
        "Days Since Previous Campaign Contact (999 = Not Previously Contacted)",
        min_value=0,
        max_value=999,
        value=999
    )

    customer_data["previous"] = st.number_input(
        "Contacts Before Current Campaign",
        min_value=0,
        max_value=100,
        value=0
    )

    customer_data["poutcome"] = st.selectbox(
        "Previous Campaign Outcome",
        categorical_options["poutcome"]
    )

    st.markdown("### Economic Indicators")

    for column in [
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed"
    ]:
        customer_data[column] = st.number_input(
            column.replace(".", " ").title(),
            value=float(numeric_defaults[column]),
            format="%.3f"
        )

    submitted = st.form_submit_button(
        "Predict Subscription"
    )

# --------------------------------------------------
# 6. Make Prediction
# --------------------------------------------------

if submitted:

    input_df = pd.DataFrame([customer_data])

    # Encode categorical features
    input_encoded = pd.get_dummies(
        input_df,
        dtype=int
    )

    # Match the exact feature columns used during training
    input_encoded = input_encoded.reindex(
        columns=model_columns,
        fill_value=0
    )

    # Predict customer subscription
    prediction = model.predict(input_encoded)[0]

    # Handle either numeric or text target labels
    is_yes = str(prediction).strip().lower() in {
        "1", "yes", "true"
    }

    st.subheader("Prediction Result")

    if is_yes:
        st.success(
            "The model predicts: Customer is likely to subscribe (YES)."
        )
    else:
        st.warning(
            "The model predicts: Customer is unlikely to subscribe (NO)."
        )

    # Display subscription probability when available
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_encoded)[0]

        classes = [
            str(label).strip().lower()
            for label in model.classes_
        ]

        yes_labels = {"1", "yes", "true"}

        yes_index = next(
            (
                index
                for index, label in enumerate(classes)
                if label in yes_labels
            ),
            None
        )

        if yes_index is not None:
            yes_probability = probabilities[yes_index]

            st.metric(
                "Estimated Subscription Probability",
                f"{yes_probability:.2%}"
            )

st.caption(
    "This prediction is based on a trained machine learning model. "
    "It is an estimate, not a guarantee of customer behavior."
)