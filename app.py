import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Cafe Analytics Command Center",
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
        color: #f8fafc;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] span, [data-testid="stSidebar"] p {
        color: #f8fafc !important;
    }
    h1, h2, h3 {
        color: #0f172a;
        font-family: 'Inter', sans-serif;
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

# --- DATA LOADING (Cached) ---
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "restaurant_orders.csv")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        df = pd.read_csv("restaurant_orders.csv")

    # Ensure correct data types
    if "City" in df.columns:
        df["City"] = df["City"].astype(str).str.strip()
    
    # Calculate total revenue per transaction
    if "Price" in df.columns and "Quantity" in df.columns:
        df["Total_Sales"] = df["Price"] * df["Quantity"]
    else:
        df["Total_Sales"] = 0.0

    return df

df = load_data()

# --- HEADER SECTION ---
st.title("☕ Cafe Data Lab — Restaurant Analytics")
st.markdown("Performance metrics, revenue distribution, and operational insights across regional branches.")
st.markdown("---")

# --- SIDEBAR FILTERS ---
st.sidebar.title("📊 Filter Options")

available_cities = sorted(df["City"].unique().tolist()) if "City" in df.columns else []
selected_cities = st.sidebar.multiselect(
    "Select Cities:",
    options=available_cities,
    default=available_cities
)

# Apply City Filter
if not selected_cities:
    st.warning("⚠️ Please select at least one city from the sidebar.")
    st.stop()

filtered_df = df[df["City"].isin(selected_cities)]

if filtered_df.empty:
    st.error("No data found for the selected cities.")
    st.stop()

# --- KPI METRICS CARDS ---
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

# --- VISUALIZATION DASHBOARD ---
st.subheader("Regional Performance & Operational Breakdown")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 1. Revenue by City")
    city_sales = filtered_df.groupby("City")["Total_Sales"].sum().reset_index().sort_values("Total_Sales", ascending=True)
    
    fig_city = px.bar(
        city_sales,
        x="Total_Sales",
        y="City",
        orientation="h",
        color="Total_Sales",
        color_continuous_scale=px.colors.sequential.Teal,
        template="plotly_white",
        labels={"Total_Sales": "Revenue ($)", "City": "City"}
    )
    fig_city.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_city, use_container_width=True)

with col2:
    st.markdown("### 2. Payment Method Breakdown")
    if "Payment Method" in filtered_df.columns:
        payment_counts = filtered_df["Payment Method"].value_counts().reset_index()
        payment_counts.columns = ["Method", "Count"]
        
        fig_pay = px.pie(
            payment_counts,
            names="Method",
            values="Count",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Sunset,
            template="plotly_white"
        )
        fig_pay.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_pay, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.markdown("### 3. Top Performing Products")
    if "Product" in filtered_df.columns:
        top_products = filtered_df.groupby("Product")["Total_Sales"].sum().reset_index().sort_values("Total_Sales", ascending=False).head(7)
        
        fig_prod = px.bar(
            top_products,
            x="Product",
            y="Total_Sales",
            color="Total_Sales",
            color_continuous_scale=px.colors.sequential.Viridis,
            template="plotly_white",
            labels={"Total_Sales": "Revenue ($)", "Product": "Item"}
        )
        fig_prod.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_prod, use_container_width=True)

with col4:
    st.markdown("### 4. Purchase Type Split")
    if "Purchase Type" in filtered_df.columns:
        type_counts = filtered_df["Purchase Type"].value_counts().reset_index()
        type_counts.columns = ["Purchase Type", "Count"]
        
        fig_type = px.pie(
            type_counts,
            names="Purchase Type",
            values="Count",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Blues_r,
            template="plotly_white"
        )
        fig_type.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_type, use_container_width=True)

st.markdown("---")

# --- RAW DATA INSPECTION ---
with st.expander("🔍 View Transactional Log Data"):
    st.dataframe(filtered_df, use_container_width=True)