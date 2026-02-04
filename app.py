import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import os

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="DiaGuard AI",
    page_icon="🏥",
    layout="wide"
)

# --------------------------------------------------
# Custom Styling
# --------------------------------------------------
st.markdown("""
<style>
.main {padding: 0.5rem 1.5rem;}
.stAlert {border-radius: 0.6rem;}
h1 {color: #1f77b4;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model & Scaler
# --------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model = joblib.load(os.path.join(base_dir, "diabetes_model.pkl"))
        scaler = joblib.load(os.path.join(base_dir, "scaler.pkl"))
        return model, scaler
    except Exception:
        return None, None

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🏥 DiaGuard AI")
st.markdown("### AI-Powered Diabetes Risk Assessment System")

with st.expander("ℹ️ How this prediction works"):
    st.markdown("""
    **Workflow**
    1. Patient medical data is entered  
    2. Inputs are standardized using training statistics  
    3. Machine Learning model evaluates diabetes risk  
    4. Risk probability & explanation are shown  
    """)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.error("❌ Model files not found!")
    st.info("Run the Jupyter notebook to generate `diabetes_model.pkl` and `scaler.pkl`.")
    st.stop()

# --------------------------------------------------
# Sidebar – Patient Inputs (Form UX)
# --------------------------------------------------
st.sidebar.title("⚙️ Patient Information")

# -------- Model Info (Read-only) --------
st.sidebar.subheader("📊 Model Info")
st.sidebar.write("Algorithm: **Random Forest**")
st.sidebar.write("Accuracy: **~91%**")
st.sidebar.write("Dataset: **PIMA Diabetes**")

st.sidebar.markdown("---")

with st.sidebar.form("input_form"):

    # ---------------- Demographics ----------------
    st.sidebar.subheader("🧑‍⚕️ Demographics")
    st.sidebar.caption("Basic patient information affecting long-term risk")

    age = st.sidebar.slider(
        "Age (years)",
        21, 100, 30,
        help="Risk of Type 2 diabetes increases after age 45"
    )

    pregnancies = st.sidebar.number_input(
        "Number of Pregnancies",
        0, 20, 0,
        help="Higher pregnancies may increase gestational diabetes risk"
    )

    st.sidebar.markdown("---")

    # ---------------- Medical Measurements ----------------
    st.sidebar.subheader("🩺 Medical Measurements")
    st.sidebar.caption("Clinical parameters used for diabetes risk estimation")

    glucose = st.sidebar.slider(
        "Plasma Glucose (mg/dL)",
        0, 200, 120,
        help="Normal fasting glucose: 70–99 mg/dL"
    )

    bp = st.sidebar.slider(
        "Blood Pressure (mm Hg)",
        0, 130, 70,
        help="Normal systolic BP: 60–80 mm Hg"
    )

    skin = st.sidebar.slider(
        "Skin Thickness (mm)",
        0, 100, 20,
        help="Skinfold thickness indicates body fat"
    )

    insulin = st.sidebar.slider(
        "Insulin Level (μU/ml)",
        0, 900, 80,
        help="Normal insulin: 16–166 μU/ml"
    )

    bmi = st.sidebar.number_input(
        "Body Mass Index (BMI)",
        10.0, 70.0, 25.0, step=0.1,
        help="Healthy BMI: 18.5–24.9"
    )

    dpf = st.sidebar.slider(
        "Diabetes Pedigree Function",
        0.0, 2.5, 0.5, step=0.01,
        help="Higher value indicates stronger genetic risk"
    )

    # ---------------- Predict Button (LAST) ----------------
    st.sidebar.markdown("---")
    predict_btn = st.sidebar.button(
    "✨ Predict Diabetes Risk",
    use_container_width=True
)

# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------
if predict_btn:
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    try:
        probs = model.predict_proba(input_scaled)[0]
        prob_non = probs[0] * 100
        prob_yes = probs[1] * 100
    except:
        prob_yes = 100 if prediction == 1 else 0
        prob_non = 100 - prob_yes

    st.markdown("---")
    st.header("🎯 Prediction Results")

    col1, col2 = st.columns([2, 1])

    # ---------------- Result Message ----------------
    with col1:
        if prob_yes < 30:
            st.success("### ✅ LOW RISK — No Diabetes Detected")
        elif prob_yes < 70:
            st.warning("### ⚠️ MODERATE RISK — Monitor Closely")
        else:
            st.error("### 🔴 HIGH RISK — Diabetes Likely")

        st.subheader("📊 Probability Breakdown")
        c1, c2 = st.columns(2)
        c1.metric("Non-Diabetic", f"{prob_non:.1f}%")
        c2.metric("Diabetic", f"{prob_yes:.1f}%")

    # ---------------- Gauge Chart ----------------
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_yes,
            number={"suffix": "%"},
            title={"text": "Diabetes Risk"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 30], "color": "lightgreen"},
                    {"range": [30, 70], "color": "khaki"},
                    {"range": [70, 100], "color": "salmon"},
                ],
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- Health Indicator Analysis ----------------
    st.markdown("---")
    st.subheader("⚖️ Health Indicator Analysis")

    risk_factors = []
    positive_factors = []

    if glucose > 125:
        risk_factors.append("🔴 High glucose level (>125 mg/dL)")
    elif glucose < 100:
        positive_factors.append("🟢 Normal glucose level")

    if bmi > 30:
        risk_factors.append("🔴 High BMI (Obesity)")
    elif 18.5 <= bmi <= 24.9:
        positive_factors.append("🟢 Healthy BMI range")

    if bp > 80:
        risk_factors.append("🔴 Elevated blood pressure")
    elif 60 <= bp <= 80:
        positive_factors.append("🟢 Normal blood pressure")

    if age > 45:
        risk_factors.append("🟡 Age-related risk factor")
    else:
        positive_factors.append("🟢 Lower age-related risk")

    if dpf > 0.5:
        risk_factors.append("🟡 Higher genetic predisposition")
    else:
        positive_factors.append("🟢 Lower genetic predisposition")

    col_risk, col_positive = st.columns(2)

    with col_risk:
        st.subheader("⚠️ Risk Factors")
        if risk_factors:
            for r in risk_factors:
                st.warning(r)
        else:
            st.success("No significant risk factors detected")

    with col_positive:
        st.subheader("✅ Positive Health Indicators")
        if positive_factors:
            for p in positive_factors:
                st.success(p)
        else:
            st.info("No strong positive indicators identified")

    # ---------------- Recommendations ----------------
    st.markdown("---")
    st.subheader("💡 Recommendations")

    if prediction == 1:
        st.error("""
        • Consult a healthcare professional  
        • Schedule a full diabetes screening  
        • Monitor blood glucose regularly  
        • Adopt lifestyle changes (diet & exercise)
        """)
    else:
        st.success("""
        • Maintain a balanced diet  
        • Exercise regularly  
        • Monitor weight and glucose levels  
        • Routine health checkups
        """)

    # ---------------- Disclaimer ----------------
    st.markdown("---")
    st.warning("""
    **Medical Disclaimer**  
    This tool is for educational purposes only and does **not** replace
    professional medical diagnosis or treatment.
    """)

# --------------------------------------------------
# Landing Page
# --------------------------------------------------
else:
    st.info("👈 Enter patient information in the sidebar and click **Predict**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "Random Forest")
    c2.metric("Accuracy", "~91%")
    c3.metric("Dataset Size", "1000 samples")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("""
© 2026 DiaGuard AI  
Educational & Research Use Only | Not a Medical Device
""")
