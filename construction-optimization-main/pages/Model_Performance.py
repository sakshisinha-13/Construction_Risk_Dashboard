import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="Model Performance", layout="centered")

# --------------------------
# Title and Introduction
# --------------------------
st.title("Model Performance Dashboard")
st.warning("🔒 This section is for internal evaluation of the prediction model. Not intended for public users.")
st.markdown("""
Gain insights into how well the AI model is performing by reviewing accuracy metrics and misclassification patterns.
""")
st.markdown("---")

# --------------------------
# Load Dataset and Model
# --------------------------
df = pd.read_csv("new_dataset.csv")
label_encoder = LabelEncoder()
df['Risk_Level'] = label_encoder.fit_transform(df['Risk_Level'])

X = df.drop(columns=['Project_ID', 'Start_Date', 'End_Date', 'Risk_Level'])
X = pd.get_dummies(X)
y = df['Risk_Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = joblib.load('risk_model.pkl')
y_pred = model.predict(X_test)

# --------------------------
# Confusion Matrix Section
# --------------------------
st.markdown("### Confusion Matrix")
st.markdown("""
The confusion matrix compares predicted vs. actual risk levels and helps identify where the model performs well 
and where it may misclassify. This is essential for assessing the quality of predictions.
""")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Risk Level")
plt.ylabel("Actual Risk Level")
plt.title("Confusion Matrix")
st.pyplot(fig)
