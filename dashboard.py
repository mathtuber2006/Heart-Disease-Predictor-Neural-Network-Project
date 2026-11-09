import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Heart Disease Predictor",
    layout="centered"
)

st.header("Heart Disease Predictor")

st.info(
    "This is not a valid medical model. "
    "This is an educational neural network project "
    "and must not be used for medical diagnosis."
)

@st.cache_resource
def load_model():
    model = np.load("heart diseases/model_v3_updated.npz",allow_pickle=True)

    return {
        "W1": model["W1"],
        "b1": model["b1"],
        "W2": model["W2"],
        "b2": model["b2"],
        "mean": model["mean"],
        "std": model["std"],
        "columns": model["columns"].tolist()
    }

model = load_model()

def Relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

st.subheader("Enter Your Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=40
    )

    sex = st.selectbox(
        "Sex",
        ["M", "F"]
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ASY", "ATA", "NAP", "TA"]
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar",
        [0, 1],
        format_func=lambda x: "Greater than 120 mg/dl" if x == 1 else "120 mg/dl or below"
    )

    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["N", "Y"]
    )

with col2:
    oldpeak = st.number_input(
        "Oldpeak",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.1
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Down", "Flat", "Up"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure",
        min_value=1,
        max_value=300,
        value=120
    )

    max_hr = st.number_input(
        "Maximum Heart Rate",
        min_value=1,
        max_value=300,
        value=150
    )

if st.button("Predict", type="primary", use_container_width=True):

    bphr_ratio = resting_bp / max_hr

    test_data = pd.DataFrame([{
        "Age": age,
        "FastingBS": fasting_bs,
        "Oldpeak": oldpeak,
        "bphr_ratio": bphr_ratio,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "ExerciseAngina": exercise_angina,
        "ST_Slope": st_slope
    }])

    my_data = pd.get_dummies(
        test_data,
        columns=[
            "Sex",
            "ChestPainType",
            "ExerciseAngina",
            "ST_Slope"
        ],
        dtype=int
    )

    my_data = my_data.reindex(
        columns=model["columns"],
        fill_value=0
    )

    continuous_cols = [
        "Age",
        "Oldpeak",
        "bphr_ratio"
    ]

    mean = model["mean"]
    std = model["std"]

    my_data[continuous_cols] = (
        my_data[continuous_cols] - mean
    ) / std

    X_new = my_data.to_numpy(dtype=float)

    z1 = np.dot(X_new, model["W1"]) + model["b1"]
    a1 = Relu(z1)

    z2 = np.dot(a1, model["W2"]) + model["b2"]
    probability = sigmoid(z2)

    threshold = 0.5
    prediction = int(probability.item() >= threshold)

    st.divider()
    st.subheader("Prediction Results")

    st.metric(
        "Predicted Probability",
        f"{probability.item():.2%}"
    )

    st.caption(f"Classification threshold: {threshold}")

    if prediction == 1:
        st.warning(
            "The model classified this input as "
            "Heart Disease (Class 1)."
        )
    else:
        st.success(
            "The model classified this input as "
            "No Heart Disease (Class 0)."
        )

    st.caption(
        "This output is an experimental model prediction, "
        "not a medical assessment. Do not use it to make "
        "healthcare decisions."
    )
