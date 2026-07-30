☕ Cafe Operations & AI Dashboard

An end-to-end Data Science, Data Engineering, and Machine Learning project that transforms raw restaurant orders into an interactive analytics dashboard and a predictive AI model.

📸 Dashboard Preview
<img width="1917" height="907" alt="Screenshot 2026-07-30 180117" src="https://github.com/user-attachments/assets/ae8d0cc2-c208-40ee-8261-a63ae390ae60" />




(Hey! When you are on GitHub, edit this file and literally drag-and-drop a screenshot of your beautiful Streamlit app right here! Delete this text once you do.)

🌟 Project Highlights

Data Engineering Pipeline: Automated Python scripts to ingest, clean, and migrate raw CSV data into a relational SQLite database.

Business Analytics: Advanced SQL queries and Pandas aggregations to uncover top products, busy cities, and payment trends.

Machine Learning: A Logistic Regression model (with Feature Engineering) that predicts a customer's payment method with 88% accuracy.

Interactive Dashboard: A premium web application built with Streamlit and Plotly for real-time data exploration and AI predictions.

🛠️ Technologies Used

Language: Python 3

Data Engineering: SQLite3, Pandas

Machine Learning: Scikit-Learn (Logistic Regression, One-Hot Encoding)

Visualization & UI: Streamlit, Plotly Express, Matplotlib

🚀 How to Run this Project Locally

1. Clone the repository and install dependencies:

pip install pandas matplotlib scikit-learn streamlit plotly


2. Setup the Database:
Run the setup script to ingest the raw data and create the local SQLite database.

python database_setup.py


3. Launch the Web Dashboard:
Start the Streamlit server to view the interactive dashboard.

python -m streamlit run app.py


🧠 Machine Learning Insights

The AI model was trained to predict whether a customer will pay with Cash or a Credit Card.

Feature Engineering: Extracted days of the week from raw dates to create an Is_Weekend indicator, significantly boosting model performance.

Data Cleaning: Scrubbed trailing/leading spaces from raw text data to prevent classification errors.

Performance: Achieved an overall accuracy of 88%, with a 1.00 Precision score for Cash transactions.






