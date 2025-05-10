# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="Construction Risk Dashboard", layout="wide")

# --------------------------
# Sidebar Instructions
# --------------------------
st.sidebar.markdown("## How to Use")
st.sidebar.write("""
Welcome to the **AI-Powered Construction Risk Monitoring Tool**.

This tool:
- Predicts risk levels using AI
- Highlights supply chain risks
- Suggests real-time mitigation tips

Steps:
1. Fill out project parameters
2. Click **Predict Risk**

**Risk Levels:**
- 🟢 Low: Project is under control
- 🟡 Medium: Monitor proactively
- 🔴 High: Immediate action needed

Recommended Users:
- Site Engineers
- Project Managers
- Civil Planning Teams
""")

# --------------------------
# Section 1: Load and Prepare Dataset
# --------------------------
st.title("AI-Powered Construction Risk Monitoring Dashboard")

# Load dataset
df = pd.read_csv("new_dataset.csv")

# Encode target variable
label_encoder = LabelEncoder()
df['Risk_Level'] = label_encoder.fit_transform(df['Risk_Level'])

# Drop non-useful columns and prepare features
X = df.drop(columns=['Project_ID', 'Start_Date', 'End_Date', 'Risk_Level'])
X = pd.get_dummies(X)
y = df['Risk_Level']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and save model
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'risk_model.pkl')

# Reload model
y_pred = model.predict(X_test)
model = joblib.load('risk_model.pkl')

# Evaluation
report_dict = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
report_str = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("\nClassification Report:\n")
print(report_str)

cm = confusion_matrix(y_test, y_pred)

# --------------------------
# Section 2: Dashboard UI
# --------------------------
st.markdown("""
This dashboard uses AI/ML to predict project risk and simulate real-time suggestions for optimized execution in civil engineering projects.
""")

# Two-column layout for inputs
st.markdown("## Enter Parameters")

with st.form("input_form"):
    st.markdown("#### Resource & Risk Inputs")
    col1, col2 = st.columns(2)

    with col1:
        material_usage = st.slider("Material Usage (kg)", min_value=0.0, max_value=1000.0, value=300.0, step=1.0)
        material_cost_per_kg = st.number_input("Material Cost per kg (₹)", min_value=1, max_value=1000, value=50)
        equipment_utilization = st.slider("Equipment Utilization (%)", 0.0, 100.0, 75.0)
        accident_count = st.number_input("Accident Count", 0, 20, 2)
        safety_risk_score = st.slider("Safety Risk Score", 0.0, 10.0, 5.0, step=0.1)
        anomaly_detected = st.radio("Anomaly Detected", [0, 1], horizontal=True)

    with col2:
        energy_consumption = st.slider("Energy Consumption (kWh)", 0.0, 100000.0, 50000.0)
        labor_hours = st.number_input("Labor Hours", 0, 20000, 8000)
        site_condition = st.selectbox("Site Condition Today", ["Normal", "Muddy", "Blocked", "Stormy", "Unstable"])
        schedule_deviation = st.slider("Schedule Deviation (%)", 0.0, 100.0, 5.0)
        cost_overrun = st.slider("Cost Overrun (%)", 0.0, 100.0, 10.0)

    st.markdown("---")
    submit = st.form_submit_button("Predict Risk", use_container_width=True)

# Prediction
if submit:
    input_df = pd.DataFrame({
        'Material_Usage': [material_usage],
        'Equipment_Utilization': [equipment_utilization],
        'Accident_Count': [accident_count],
        'Safety_Risk_Score': [safety_risk_score],
        'Anomaly_Detected': [anomaly_detected],
        'Energy_Consumption': [energy_consumption],
        'Labor_Hours': [labor_hours],
        'Site_Condition': [site_condition],
        'Schedule_Deviation': [schedule_deviation],
        'Cost_Overrun': [cost_overrun]
    })

    input_df = pd.get_dummies(input_df)

    # 🚦 AUTOMATED FLAGGING BASED ON INPUTS
    risk_factors = []

    if schedule_deviation > 20:
        risk_factors.append("High Schedule Delay Detected")

    if cost_overrun > 25:
        risk_factors.append("Cost Overrun Beyond Threshold")

    if anomaly_detected == 1:
        risk_factors.append("Anomaly Reported in Project")

    if equipment_utilization < 50:
        risk_factors.append("Low Equipment Utilization")

    if len(risk_factors) > 0:
        st.markdown("Automated Issue Flags")
        for flag in risk_factors:
            st.warning(flag)
    else:
        st.success("No critical flags raised — system shows healthy project indicators.")

    # Supply Chain Risk Scoring
    supply_risk_score = (
        schedule_deviation * 0.5 +
        cost_overrun * 0.4 +
        anomaly_detected * 10 * 0.1
    )

    if supply_risk_score >= 7:
        st.warning("High supply chain risk detected based on schedule delays, cost overruns, and anomalies.")
    elif supply_risk_score >= 4:
        st.info("Moderate supply chain risk detected.")
    else:
        st.success("Supply chain appears stable.")

    # Reindex to match training features
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    if all(value == 0 for value in input_df.iloc[0]):
        st.warning("⚠️ All input values are zero. Please provide valid project data.")
    else:
        # Estimated material cost
        estimated_material_cost = material_usage * material_cost_per_kg
        st.info(f"Estimated Material Cost: ₹{estimated_material_cost:,.2f}")

        # Risk prediction
        prediction = model.predict(input_df)[0]
        risk_label = label_encoder.inverse_transform([prediction])[0]

        st.markdown("## Prediction Result")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("Predicted Risk Level", risk_label.upper())

            estimated_material_cost = material_usage * material_cost_per_kg
            st.metric("Material Cost (₹)", f"{estimated_material_cost:,.2f}")

        with col2:
            st.markdown("### Supply Chain Risk Level")
            if supply_risk_score >= 7:
                st.warning("High supply chain risk detected (Schedule delays, Cost overrun, Anomalies).")
            elif supply_risk_score >= 4:
                st.info("Moderate supply chain risk.")
            else:
                st.success("Supply chain is stable.")

        st.divider()

        # Mitigation
        st.markdown("### Recommended Mitigation Actions")
        mitigations = []
        if schedule_deviation > 20:
            mitigations.append("Reallocate resources or extend project timeline.")
        if cost_overrun > 25:
            mitigations.append("Reassess procurement contracts and control budget leakages.")
        if accident_count > 3:
            mitigations.append("Conduct safety training and add on-site safety supervisors.")
        if anomaly_detected == 1:
            mitigations.append("Investigate anomaly cause and enhance monitoring.")
        if equipment_utilization < 50:
            mitigations.append("Optimize equipment scheduling or inspect idle machinery.")
        if energy_consumption > 80000:
            mitigations.append("Audit equipment for energy efficiency.")

        for tip in mitigations:
            st.info(tip)
        if not mitigations:
            st.success("No major issues detected. Maintain current strategy.")

        st.divider()

        # Cost/Resource Efficiency
        st.markdown("### Cost & Resource Efficiency Insights")
        if material_usage < 200:
            st.info("Material usage is efficient.")
        if equipment_utilization >= 80:
            st.success("Equipment utilization is optimal.")
        if labor_hours > 18000:
            st.warning("High labor hours. Consider optimizing team size.")
        if energy_consumption < 40000:
            st.success("Energy use is within efficient limits.")
        if cost_overrun <= 5:
            st.success("Project is within planned budget.")

        st.divider()

        # Post-Mitigation Estimates
        st.markdown("### Estimated Post-Mitigation Usage")
        estimated_material_usage = material_usage * 0.95 if cost_overrun > 25 else material_usage
        estimated_cost = estimated_material_usage * material_cost_per_kg
        estimated_labor_hours = labor_hours * 0.9 if labor_hours > 18000 else labor_hours
        estimated_energy = energy_consumption * 0.9 if energy_consumption > 80000 else energy_consumption

        st.write(f"**Adjusted Material Usage:** {estimated_material_usage:.2f} kg")
        st.write(f"**Post-Mitigation Material Cost:** ₹{estimated_cost:,.2f}")
        st.write(f"**Adjusted Labor Hours:** {estimated_labor_hours:.0f}")
        st.write(f"**Estimated Energy Consumption:** {estimated_energy:.0f} kWh")

        st.markdown("---")