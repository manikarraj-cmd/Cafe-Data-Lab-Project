# ☕ Cafe Data Lab — Executive Analytics & Revenue Prediction Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Model-Random_Forest-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Visualization-Plotly-purple?logo=plotly&logoColor=white)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, production-ready machine learning framework and executive decision dashboard designed to forecast restaurant transaction revenue and analyze regional branch operations in real time.

---

## 📌 Executive Summary & Business Value

In retail food and beverage operations, optimizing basket size and anticipating order revenue across regional locations is critical for inventory forecasting, staffing, and dynamic cross-selling.

This framework leverages historical transactional logs across regional branches (London, Madrid, Lisbon, Berlin, Paris) to fit an ensemble **Random Forest Regressor**, enabling operational teams and store managers to predict transaction monetary values based on early checkout parameters.

### 🎯 Key Strategic Use Cases
* **Real-time Order Value Forecasting:** Input order parameters (unit price, item volume, fulfillment type, branch city) to estimate total revenue before settlement.
* **Smart Upselling & Loyalty Tiering:** Automatically trigger VIP loyalty enrollment or targeted promo suggestions based on predicted order tiers.
* **Regional Operational Diagnostics:** Track gross revenue, unit sales volume, payment channel breakdown, and top-performing menu products across all active branches.

---

## 🏗️ System Architecture & Data Pipeline

┌─────────────────────────────────┐
│ Raw Transaction Receipts        │ (Order ID, Product, Price, Quantity, City, Payment)
└────────────────┬────────────────┘
│
▼
┌─────────────────────────────────┐
│ Data Preprocessing & Cleaning   │ String stripping, handling missing values, Total_Sales computation
└────────────────┬────────────────┘
│
▼
┌─────────────────────────────────┐
│ One-Hot Feature Encoding        │ Categorical vectorization (Purchase Type, Payment Method, Branch City)
└────────────────┬────────────────┘
│
▼
┌─────────────────────────────────┐
│ Random Forest Regression Engine │ Ensemble model fitting & cross-validated R² evaluation
└────────────────┬────────────────┘
│
▼
┌─────────────────────────────────┐
│ Executive Streamlit Interface   │ Real-time decision analytics dashboard & scenario simulator
└─────────────────────────────────┘


---

## 📊 Feature Matrix & Model Architecture

Raw order logs are transformed into model-ready features capturing operational and monetary attributes:

| Feature Dimension | Variable Name | Type | Business Rationale |
| :--- | :--- | :--- | :--- |
| **Unit Economics** | `Price` | Numerical | Baseline price threshold per menu item. |
| **Basket Volume** | `Quantity` | Numerical | Unit count ordered per transaction batch. |
| **Fulfillment Mode** | `Purchase Type` | Categorical | Channel dynamics (Dine-In vs. Takeaway). |
| **Payment Channel** | `Payment Method` | Categorical | Payment settlement preference (Credit Card, Cash, Digital Wallet). |
| **Geographic Location**| `City` | Categorical | Branch performance metrics across European regional markets. |
| **Target Variable** | `Total_Sales` | Numerical | Computed total transaction monetary value (`Price` × `Quantity`). |

### 🧠 Model Performance & Hyperparameters
* **Primary Model:** Random Forest Regressor (`n_estimators=150`, `random_state=42`)
* **Validation Strategy:** Holdout Train-Test Split (80/20 ratio)
* **Model Evaluation Metric:** Coefficient of Determination ($R^2$ Score)

---

## 💻 Dashboard Capabilities

The Streamlit executive control center provides three distinct functional views:

### 1. 📊 Executive Performance Overview
* **KPI Metrics:** Real-time metrics for Gross Revenue, Total Order Volume, Average Order Value (AOV), and Total Units Sold.
* **Branch Revenue Breakdown:** Horizontal bar charts displaying revenue contribution by city.
* **Payment Settlement Distribution:** Interactive donut chart analyzing transaction channel split.
* **Menu Product Performance:** Volume and revenue breakdown for top-selling items.

### 2. 🔮 AI Revenue Scenario Simulator
Interactive scenario tool where branch managers adjust order parameters via UI sliders to generate instant AI revenue forecasts accompanied by automated strategic business recommendations.

### 3. 📜 Transaction Logs Explorer
Tabular data browser providing real-time data inspection and filtering capabilities.

---

## 🚀 Quick Start & Local Deployment

### Prerequisites
* Python 3.10+
* Git

### Installation & Execution Steps

1. **Clone Repository:**
   ```bash
   git clone [https://github.com/manikarraj-cmd/cafe-data-lab-project.git](https://github.com/manikarraj-cmd/cafe-data-lab-project.git)
   cd cafe-data-lab-project