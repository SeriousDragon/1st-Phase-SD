import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Predictor")
st.caption("Введите признаки пациента и получите предсказание (0 = нет болезни, 1 = есть болезнь).")

# ---- Load trained pipeline ----
@st.cache_resource
def load_pipeline(path: str = "best_pipeline.pkl"):
    p = Path(path)
    if not p.exists():
        st.error(
            "Файл 'best_pipeline.pkl' не найден. "
            "Сначала обучите пайплайн и сохраните его:\n\n"
            "```python\n"
            "import joblib\n"
            "joblib.dump(best_pipeline, 'best_pipeline.pkl')\n"
            "```"
        )
        return None
    return joblib.load(p)

pipe = load_pipeline()

# Feature lists (должны совпадать с обучением)
numeric_features = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
categorical_features = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
all_features = numeric_features + categorical_features

st.subheader("Ввод данных")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (лет)", min_value=0, max_value=120, value=50, step=1)
        resting_bp = st.number_input("RestingBP", min_value=0, max_value=300, value=130, step=1)
        cholesterol = st.number_input("Cholesterol", min_value=0, max_value=1000, value=200, step=1)
        fasting_bs = st.selectbox("FastingBS (>120 mg/dl?)", options=[0, 1], index=0)

    with col2:
        max_hr = st.number_input("MaxHR", min_value=60, max_value=220, value=150, step=1)
        oldpeak = st.number_input("Oldpeak (ST depression)", min_value=-5.0, max_value=10.0, value=1.0, step=0.1)
        sex = st.selectbox("Sex", options=["M", "F"], index=0)
        chest_pain = st.selectbox("ChestPainType", options=["TA", "ATA", "NAP", "ASY"], index=1)

    resting_ecg = st.selectbox("RestingECG", options=["Normal", "ST", "LVH"], index=0)
    exercise_angina = st.selectbox("ExerciseAngina", options=["N", "Y"], index=0)
    st_slope = st.selectbox("ST_Slope", options=["Up", "Flat", "Down"], index=1)

    submitted = st.form_submit_button("Предсказать")

if submitted:
    sample = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingECG": resting_ecg,
        "ExerciseAngina": exercise_angina,
        "ST_Slope": st_slope,
    }
    X_input = pd.DataFrame([sample], columns=all_features)

    if pipe is not None:
        y_pred = pipe.predict(X_input)[0]
        st.success(f"Предсказанный класс: **{int(y_pred)}** (0=нет болезни, 1=болезнь)")

        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_input)[0][1]
            st.info(f"Вероятность болезни сердца: **{proba:.2%}**")

        st.markdown("### Введённые данные")
        st.dataframe(X_input)
