import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Cafe Analytics & Revenue Prediction Engine",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] span, [data-testid="stSidebar"] p {
        color: #f8fafc !important;
    }
    div[data-testid="stMetric"] {
        background-color: #1e293b !important;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stMetricLabel"] label {
        color: #94a3b8 !important;
    }
    div[data-testid="stMetricValue"] div {
        color: #38bdf8 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- DATA LOADING & PREPROCESSING (Cached) ---
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
    
    # Calculate revenue feature
    if "Price" in df.columns and "Quantity" in df.columns:
        df["Total_Sales"] = df["Price"] * df["Quantity"]
    else:
        df["Total_Sales"] = 0.0

    return df

df = load_data()

# --- TRAIN ML PREDICTION MODEL ---
@st.cache_resource
def train_clv_model(data):
    features = ["Price", "Quantity", "Purchase Type", "Payment Method", "City"]
    available_features = [col for col in features if col in data.columns]
    
    # Prepare encoded dataset for Random Forest model
    df_model = data[available_features + ["Total_Sales"]].dropna()
    X = pd.get_dummies(df_model[available_features], drop_first=True)
    y = df_model["Total_Sales"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_state=42, test_size=0.2)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else 1.0

    return model, X.columns.tolist(), r2

model, feature_columns, model_r2 = train_clv_model(df)

# --- NAVIGATION & SIDEBAR ---
st.sidebar.title("☕ Navigation")
app_mode = st.sidebar.radio("Select View:", ["📊 Executive Analytics Dashboard", "🔮 Revenue Prediction Engine"])

# --- VIEW 1: EXECUTIVE ANALYTICS DASHBOARD ---
if app_mode == "📊 Executive Analytics Dashboard":
    st.title("☕ Cafe Data Lab — Restaurant Analytics")
    st.markdown("Performance metrics, revenue distribution, and operational insights across regional branches.")
    st.markdown("---")

    available_cities = sorted(df["City"].unique().tolist()) if "City" in df.columns else []
    selected_cities = st.sidebar.multiselect(
        "Filter by City:",
        options=available_cities,
        default=available_cities
    )

    if not selected_cities:
        st.warning("⚠️ Please select at least one city from the sidebar.")
        st.stop()

    filtered_df = df[df["City"].isin(selected_cities)]

    # Metrics
    total_revenue = filtered_df["Total_Sales"].sum()
    total_orders = len(filtered_df)
    avg_order_val = filtered_df["Total_Sales"].mean() if total_orders > 0 else 0
    total_items = filtered_df["Quantity"].sum() if "Quantity" in filtered_df.columns else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Revenue", f"${total_revenue:,.2f}")
    m2.metric("Total Orders", f"{total_orders:,}")
    m3.metric("Avg Order Value", f"${avg_order_val:,.2f}")
    m4.metric("Items Sold", f"{total_items:,}")

    st.markdown("---")

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Revenue by City")
        city_sales = filtered_df.groupby("City")["Total_Sales"].sum().reset_index().sort_values("Total_Sales", ascending=True)
        fig_city = px.bar(city_sales, x="Total_Sales", y="City", orientation="h", color="Total_Sales", color_continuous_scale=px.colors.sequential.Teal)
        st.plotly_chart(fig_city, use_container_width=True)

    with col2:
        st.markdown("### Payment Method Breakdown")
        if "Payment Method" in filtered_df.columns:
            payment_counts = filtered_df["Payment Method"].value_counts().reset_index()
            payment_counts.columns = ["Method", "Count"]
            fig_pay = px.pie(payment_counts, names="Method", values="Count", hole=0.4, color_discrete_sequence=px.colors.sequential.Sunset)
            st.plotly_chart(fig_pay, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Top Performing Products")
        if "Product" in filtered_df.columns:
            top_products = filtered_df.groupby("Product")["Total_Sales"].sum().reset_index().sort_values("Total_Sales", ascending=False).head(7)
            fig_prod = px.bar(top_products, x="Product", y="Total_Sales", color="Total_Sales", color_continuous_scale=px.colors.sequential.Viridis)
            st.plotly_chart(fig_prod, use_container_width=True)

    with col4:
        st.markdown("### Purchase Type Split")
        if "Purchase Type" in filtered_df.columns:
            type_counts = filtered_df["Purchase Type"].value_counts().reset_index()
            type_counts.columns = ["Purchase Type", "Count"]
            fig_type = px.pie(type_counts, names="Purchase Type", values="Count", hole=0.4, color_discrete_sequence=px.colors.sequential.Blues_r)
            st.plotly_chart(fig_type, use_container_width=True)

# --- VIEW 2: REVENUE PREDICTION ENGINE ---
else:
    st.title("🔮 Order Revenue Machine Learning Predictor")
    st.markdown("Input prospective order parameters below to forecast transaction monetary value using a Random Forest Regressor.")
    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("⚙️ Simulation Inputs")
        
        input_price = st.slider("Item Unit Price ($)", float(df["Price"].min()), float(df["Price"].max()), float(df["Price"].median()))
        input_quantity = st.slider("Order Quantity", int(df["Quantity"].min()), int(df["Quantity"].max()), 2)
        input_city = st.selectbox("Target City Branch", df["City"].unique().tolist())
        input_type = st.selectbox("Purchase Type", df["Purchase Type"].unique().tolist() if "Purchase Type" in df.columns else ["Dine-In"])
        input_payment = st.selectbox("Payment Method", df["Payment Method"].unique().tolist() if "Payment Method" in df.columns else ["Credit Card"])

        predict_btn = st.button("🚀 Calculate Expected Revenue", use_container_width=True)

    with c2:
        st.subheader("🎯 Predictive Output")
        
        if predict_btn:
            # Construct feature row for prediction
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

            st.metric("Forecasted Transaction Value", f"${prediction:,.2f}")
            st.info(f"💡 **Model Confidence (R² Score):** {model_r2:.2f}")

            # Business Recommendation Tier
            if prediction > 100:
                st.success("🌟 **High-Value Order:** Qualifies for instant VIP loyalty rewards!")
            elif prediction > 50:
                st.warning("⚡ **Standard Order:** High potential for cross-selling add-on items.")
            else:
                st.error("🔹 **Low-Value Order:** Consider bundling with promo offers.")