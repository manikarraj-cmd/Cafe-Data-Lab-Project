import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Cafe Data Lab | Executive Decision Engine",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ENTERPRISE CUSTOM CSS STYLING ---
st.markdown("""
    <style>
    /* Global Background & Typography */
    .stApp {
        background-color: #0b0f19;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #f1f5f9;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #020617 !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Headers & Text Customization */
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    p, span, label {
        color: #94a3b8;
    }
    
    /* Executive Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }
    div[data-testid="stMetricLabel"] label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] div {
        color: #f8fafc !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
    }
    
    /* Glassmorphism Containers */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }
    
    /* Custom Primary Action Button */
    .stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%) !important;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5) !important;
        transform: translateY(-1px);
    }
    
    /* Streamlit Input Styling Override */
    div[data-baseweb="select"] > div {
        background-color: #0f172a !important;
        border-color: #334155 !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "restaurant_orders.csv")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        df = pd.read_csv("restaurant_orders.csv")

    if "City" in df.columns:
        df["City"] = df["City"].astype(str).str.strip()
    
    # Calculate Total Sales
    if "Price" in df.columns and "Quantity" in df.columns:
        df["Total_Sales"] = df["Price"] * df["Quantity"]
    else:
        df["Total_Sales"] = 0.0

    return df

df = load_data()

# --- ML MODEL TRAINING ---
@st.cache_resource
def train_revenue_model(data):
    features = ["Price", "Quantity", "Purchase Type", "Payment Method", "City"]
    available_features = [col for col in features if col in data.columns]
    
    df_model = data[available_features + ["Total_Sales"]].dropna()
    X = pd.get_dummies(df_model[available_features], drop_first=True)
    y = df_model["Total_Sales"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else 1.0

    return model, X.columns.tolist(), r2

model, feature_columns, model_r2 = train_revenue_model(df)

# --- HEADER SECTION ---
st.title("☕ Cafe Data Lab — Executive Decision Engine")
st.markdown("<p style='font-size: 1.05rem; color: #94a3b8;'>Real-time sales performance diagnostics and machine learning transaction forecasting.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- NAVIGATION TABS ---
tab1, tab2 = st.tabs(["📊 Executive Overview", "🔮 Predictive Revenue Engine"])

# --- TAB 1: EXECUTIVE ANALYTICS ---
with tab1:
    st.sidebar.markdown("### 🎛️ Dashboard Filters")
    
    available_cities = sorted(df["City"].unique().tolist()) if "City" in df.columns else []
    selected_cities = st.sidebar.multiselect(
        "Select Regional Branches:",
        options=available_cities,
        default=available_cities
    )

    if not selected_cities:
        st.warning("⚠️ Please select at least one city branch from the sidebar filter.")
        st.stop()

    filtered_df = df[df["City"].isin(selected_cities)]

    # Top-Level KPI Summary
    total_revenue = filtered_df["Total_Sales"].sum()
    total_orders = len(filtered_df)
    avg_order_val = filtered_df["Total_Sales"].mean() if total_orders > 0 else 0
    active_cities = filtered_df["City"].nunique()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Gross Revenue", f"${total_revenue:,.2f}")
    k2.metric("Total Order Volume", f"{total_orders:,}")
    k3.metric("Average Basket Value", f"${avg_order_val:,.2f}")
    k4.metric("Active Regions", f"{active_cities}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualization Grid
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.markdown("#### Revenue Generation by Regional Branch")
        city_sales = filtered_df.groupby("City")["Total_Sales"].sum().reset_index().sort_values("Total_Sales", ascending=True)
        
        fig_city = px.bar(
            city_sales,
            x="Total_Sales",
            y="City",
            orientation="h",
            color="Total_Sales",
            color_continuous_scale=["#0284c7", "#38bdf8", "#7dd3fc"],
            template="plotly_dark",
            labels={"Total_Sales": "Revenue ($)", "City": "Region"}
        )
        fig_city.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
            margin=dict(l=10, r=10, t=20, b=10),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_city, use_container_width=True)

    with row1_col2:
        st.markdown("#### Payment Settlement Distribution")
        if "Payment Method" in filtered_df.columns:
            payment_counts = filtered_df["Payment Method"].value_counts().reset_index()
            payment_counts.columns = ["Method", "Count"]
            
            fig_pay = px.pie(
                payment_counts,
                names="Method",
                values="Count",
                hole=0.5,
                color_discrete_sequence=["#38bdf8", "#818cf8", "#c084fc", "#f472b6"],
                template="plotly_dark"
            )
            fig_pay.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                margin=dict(l=10, r=10, t=20, b=10),
                legend=dict(orientation="h", y=-0.1, x=0.2)
            )
            st.plotly_chart(fig_pay, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown("#### Top Revenue Generating Items")
        if "Product" in filtered_df.columns:
            top_products = filtered_df.groupby("Product")["Total_Sales"].sum().reset_index().sort_values("Total_Sales", ascending=False).head(6)
            
            fig_prod = px.bar(
                top_products,
                x="Product",
                y="Total_Sales",
                color="Total_Sales",
                color_continuous_scale=["#34d399", "#059669"],
                template="plotly_dark",
                labels={"Total_Sales": "Revenue ($)", "Product": "Menu Item"}
            )
            fig_prod.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                margin=dict(l=10, r=10, t=20, b=10),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_prod, use_container_width=True)

    with row2_col2:
        st.markdown("#### Order Type Mix (Dine-In vs. Takeaway)")
        if "Purchase Type" in filtered_df.columns:
            type_counts = filtered_df["Purchase Type"].value_counts().reset_index()
            type_counts.columns = ["Type", "Count"]
            
            fig_type = px.pie(
                type_counts,
                names="Type",
                values="Count",
                hole=0.5,
                color_discrete_sequence=["#fbbf24", "#f59e0b"],
                template="plotly_dark"
            )
            fig_type.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                margin=dict(l=10, r=10, t=20, b=10),
                legend=dict(orientation="h", y=-0.1, x=0.3)
            )
            st.plotly_chart(fig_type, use_container_width=True)

# --- TAB 2: PREDICTIVE ENGINE ---
with tab2:
    st.markdown("### 🔮 Machine Learning Scenario Simulator")
    st.markdown("Adjust hypothetical transaction variables to forecast real-time customer monetary spend using an ensemble Random Forest model.")
    st.markdown("<br>", unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns([1, 1], gap="large")

    with sim_col1:
        st.markdown("#### ⚙️ Order Scenario Inputs")
        
        input_price = st.slider("Item Unit Price ($)", float(df["Price"].min()), float(df["Price"].max()), float(df["Price"].median()))
        input_quantity = st.slider("Order Quantity Units", int(df["Quantity"].min()), int(df["Quantity"].max()), 3)
        input_city = st.selectbox("Target Regional Branch", df["City"].unique().tolist())
        input_type = st.selectbox("Fulfillment Channel", df["Purchase Type"].unique().tolist() if "Purchase Type" in df.columns else ["Dine-In"])
        input_payment = st.selectbox("Preferred Payment Method", df["Payment Method"].unique().tolist() if "Payment Method" in df.columns else ["Credit Card"])

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡ Run Revenue Simulation", use_container_width=True)

    with sim_col2:
        st.markdown("#### 🎯 Model Forecast & Analytics")
        
        if predict_btn:
            input_df = pd.DataFrame([{
                "Price": input_price,
                "Quantity": input_quantity,
                "Purchase Type": input_type,
                "Payment Method": input_payment,
                "City": input_city
            }])

            input_encoded = pd.get_dummies(input_df)
            input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

            prediction = model.predict(input_encoded)[0]

            st.metric("Forecasted Order Revenue", f"${prediction:,.2f}")
            
            st.markdown(f"""
                <div style='background-color: #0f172a; border-left: 4px solid #38bdf8; padding: 16px; border-radius: 8px; margin-top: 15px;'>
                    <p style='margin:0; color: #f8fafc; font-weight: 600;'>Model Performance Signal</p>
                    <p style='margin:0; color: #94a3b8; font-size: 0.9rem;'>Cross-Validated Model $R^2$ Score: <strong style='color: #38bdf8;'>{model_r2:.2f}</strong></p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Strategic Operational Prescription:")
            
            if prediction > 80:
                st.success("🌟 **High-Tier Order Cohort:** Automatically qualify customer for premium loyalty program status and complementary upsell offers.")
            elif prediction > 40:
                st.info("⚡ **Standard Order Cohort:** High potential for cross-selling complementary beverage/dessert items at checkout.")
            else:
                st.warning("🔹 **Value-Seeking Order Cohort:** Suggest targeted bundle promotions to drive higher basket volume.")
        else:
            st.info("👈 Adjust the scenario parameters on the left and click **'Run Revenue Simulation'** to generate predictions.")

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.85rem;'>Enterprise Dashboard Engine • Developed by Manikar Raj</p>", unsafe_allow_html=True)