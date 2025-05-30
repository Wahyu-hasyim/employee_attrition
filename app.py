import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load model dan scaler
model = load("model_rf.joblib")
scaler = load("scaler.joblib")

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Karyawan Resign", layout="centered")
st.title("Prediksi Karyawan yang Akan Resign")

st.markdown("Masukkan data karyawan untuk memprediksi kemungkinan resign:")

# === Form Input ===
with st.form("form_predict"):
    col1, col2 = st.columns(2)
    with col1:
        length_of_service = st.slider("Lama Kerja (tahun)", 0, 40, 5)
        age = st.slider("Umur", 18, 65, 30)
        business_unit = st.selectbox("Unit Bisnis", ["HEADOFFICE", "STORES"])
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    with col2:
        department_risk = st.selectbox("Departemen Risiko Tinggi?", ["Ya", "Tidak"])
        city_risk = st.selectbox("Kota Risiko Tinggi?", ["Valemont", "Lainnya"])
        job_risk = st.selectbox("Posisi Risiko Tinggi?", ["VP Stores", "Lainnya"])

    submitted = st.form_submit_button("🔍 Prediksi")

if submitted:
    # Mapping variabel
    dept_risk = 1 if department_risk == "Ya" else 0
    city_risk_val = 2 if city_risk == "Valemont" else 1
    job_risk_val = 2 if job_risk == "VP Stores" else 1
    usia_risk = 1 if 8 <= age <= 26 else 0

    bu_head = 1 if business_unit == "HEADOFFICE" else 0
    bu_store = 1 if business_unit == "STORES" else 0
    gender_m = 1 if gender == "Male" else 0
    gender_f = 1 if gender == "Female" else 0

    # Buat DataFrame input
    data_input = pd.DataFrame([{
        "length_of_service": length_of_service,
        "age": age,
        "dept_risk": dept_risk,
        "city_risk": city_risk_val,
        "job_risk": job_risk_val,
        "usia_risk": usia_risk,
        "BUSINESS_UNIT_HEADOFFICE": bu_head,
        "BUSINESS_UNIT_STORES": bu_store,
        "gender_full_Female": gender_f,
        "gender_full_Male": gender_m
    }])

    # Scaling (harus sesuai training)
    data_input["length_of_service"] = scaler.transform(data_input[["length_of_service"]])

    # ✅ Samakan urutan kolom dengan saat training
    data_input = data_input[model.feature_names_in_]

    # Prediksi
    prediction = model.predict(data_input)[0]
    proba = model.predict_proba(data_input)[0][1]

    # Output
    st.subheader("Hasil Prediksi:")
    if prediction == 1:
        st.error(f"Karyawan **berisiko resign** dengan kemungkinan {proba:.2%}")
    else:
        st.success(f"Karyawan **kemungkinan bertahan** dengan kemungkinan {(1 - proba):.2%}")
