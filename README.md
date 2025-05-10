AI-Powered Construction Risk Monitoring Dashboard
This is a Streamlit-based web application that uses AI and machine learning to predict and monitor risk levels in civil engineering and construction projects. It provides real-time insights and actionable suggestions to improve project execution.

Live Demo:
https://constructionriskdashboard-fhfthsx5etlr2womdmdguw.streamlit.app/

Features
Predicts construction project risk levels (Low, Medium, High)

Highlights key risks like cost overrun, schedule delays, and anomalies

Suggests mitigation strategies for high-risk inputs

Analyzes cost and resource efficiency

Displays post-mitigation estimations

Includes a model performance tab with confusion matrix

Project Structure
bash
Copy
Edit
.
├── Dashboard.py              # Main dashboard UI
├── pages/
│   └── Model_Performance.py # Shows confusion matrix and accuracy
├── new_dataset.csv          # Dataset used to train and test the model
├── risk_model.pkl           # Trained ML model (RandomForest)
├── requirements.txt         # Python dependencies
├── runtime.txt              # Runtime config for Streamlit Cloud
Tech Stack
Python

Streamlit

Scikit-learn

Pandas, NumPy

Matplotlib, Seaborn

How to Run Locally
Clone the repository

bash
Copy
Edit
git clone https://github.com/sakshisinha-13/Construction_Risk_Dashboard.git
cd Construction_Risk_Dashboard
Create and activate virtual environment

bash
Copy
Edit
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On macOS/Linux
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the app

bash
Copy
Edit
streamlit run Dashboard.py
Deployment
This app is deployed using Streamlit Cloud. To deploy your own version:

Push the project to a GitHub repository.

Go to Streamlit Cloud and connect your GitHub account.

Select the repository and set Dashboard.py as the main file.

Click "Deploy".

License
This project is for educational and demonstration purposes.

