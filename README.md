AI-Powered Construction Risk Monitoring Dashboard
This Streamlit-based web application uses AI and machine learning to predict and monitor risk levels in civil engineering and construction projects. It provides real-time insights, supply chain risk scoring, mitigation suggestions, and resource efficiency analysis to support better project execution.

🔗 Live Demo: https://constructionriskdashboard-fhfthsx5etlr2womdmdguw.streamlit.app/

Features
Predicts construction project risk levels: Low, Medium, or High
Highlights issues like cost overrun, schedule delays, and anomalies
Suggests rule-based mitigation strategies
Analyzes cost and resource efficiency
Displays estimated post-mitigation cost, labor, and energy usage
Dedicated Model Performance page with confusion matrix and accuracy report

Project Structure:
.
├── Dashboard.py              # Main Streamlit dashboard
├── pages/
│   └── Model_Performance.py # Model evaluation metrics (confusion matrix)
├── new_dataset.csv          # Dataset used for training and evaluation
├── risk_model.pkl           # Trained RandomForest model
├── requirements.txt         # Python package dependencies
├── runtime.txt              # Runtime configuration for Streamlit Cloud

Tech Stack
Python
Streamlit
Scikit-learn
Pandas, NumPy
Matplotlib, Seaborn

-----------------------------------------------------------------------------


