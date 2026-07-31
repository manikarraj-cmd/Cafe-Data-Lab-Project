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
    page_title="Cafe Data Lab ☕ | Executive Intelligence Engine",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ENTERPRISE UI STYLING & EMOJI BADGES ---
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background-color: #0d1117;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #e6edf3;
    }
    
    /* Sidebar Redesign */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    
    /* Glassmorphism Card Containers */
    .metric-card {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.9) 0%, rgba(13, 17, 23, 0.8) 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #30363d;
        transform: translateY(-2px);
    }
    
    /* Executive Dark Top Banner (Blue Paint Removed) */
    .header-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px 30px;
        margin-bottom: 25px;
        color: #f0f6fc;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Metric Typography */
    .metric-title {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.9rem;
        color: #f0f6fc;
        font-weight: 800;
    }
    
    /* Custom Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #161b22;
        border-radius: 8px 8px 0 0;
        border: 1px solid #30363d;
        color: #8b949e;
        font-weight: 600;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        color: #f0f6fc !important;
        border-color: #8b949e !important;
    }
    
    /* Custom Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 14px 28px !important;
        box-shadow: 0 4px 14px rgba(46, 160, 67, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2ea043 0%, #3fb950 100%) !important;
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
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
    
    if "Price" in df.columns and "Quantity" in df.columns:
        df["Total_Sales"] = df["Price"] * df["Quantity"]
    else:
        df["Total_Sales"] = 0.0

    return df

df = load_data()

# --- MODEL TRAINING ---
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

# --- CLEAN HEADER BANNER (NO BLUE PAINT) ---
st.markdown("""
    <div class="header-box">
        <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #f0f6fc;">☕ Cafe Data Lab — Executive Command Center</h1>
        <p style="margin: 5px 0 0 0; font-size: 1.05rem; color: #8b949e;">Real-time Business Intelligence, Multi-Branch Analytics & AI Revenue Forecasting</p>
    </div>
""", unsafe_allow_html=True)

# --- NAVIGATION TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Performance Dashboard", "🔮 Revenue Prediction Engine", "📜 Transactional Data Explorer"])

# --- TAB 1: EXECUTIVE ANALYTICS ---
with tab1:
    st.sidebar.markdown("### 🎛️ Regional Filters")
    
    available_cities = sorted(df["City"].unique().tolist()) if "City" in df.columns else []
    selected_cities = st.sidebar.multiselect(
        "📍 Select Branches:",
        options=available_cities,
        default=available_cities
    )

    if not selected_cities:
        st.warning("⚠️ Please select at least one city branch from the sidebar filter.")
        st.stop()

    filtered_df = df[df["City"].isin(selected_cities)]

    # KPI Metrics Banner
    total_revenue = filtered_df["Total_Sales"].sum()
    total_orders = len(filtered_df)
    avg_order_val = filtered_df["Total_Sales"].mean() if total_orders > 0 else 0
    total_items = filtered_df["Quantity"].sum() if "Quantity" in filtered_df.columns else 0

    k1, k2, k3, k4 = st.columns(4)
    
    k1.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">💰 Gross Revenue</div>
            <div class="metric-value">${total_revenue:,.2f}</div>
        </div>
    """, unsafe_allow_html=True)

    k2.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🧾 Total Orders</div>
            <div class="metric-value">{total_orders:,}</div>
        </div>
    """, unsafe_allow_html=True)

    k3.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📈 Avg Order Value</div>
            <div class="metric-value">${avg_order_val:,.2f}</div>
        </div>
    """, unsafe_allow_html=True)

    k4.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📦 Total Units Sold</div>
            <div class="metric-value">{total_items:,}</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Grid
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.markdown("### 🌆 Revenue Generation by City Branch")
        city_sales = filtered_df.groupby("City")["Total_Sales"].sum().reset_index().sort_values("Total_Sales", ascending=True)
        
        fig_city = px.bar(
            city_sales,
            x="Total_Sales",
            y="City",
            orientation="h",
            color="Total_Sales",
            color_continuous_scale=px.colors.sequential.Darkmint,
            template="plotly_dark",
            labels={"Total_Sales": "Revenue ($)", "City": "Branch Location"}
        )
        fig_city.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350, coloraxis_showscale=False)
        st.plotly_chart(fig_city, use_container_width=True)

    with row1_col2:
        st.markdown("### 💳 Payment Settlement Methods")
        if "Payment Method" in filtered_df.columns:
            payment_counts = filtered_df["Payment Method"].value_counts().reset_index()
            payment_counts.columns = ["Method", "Count"]
            
            fig_pay = px.pie(
                payment_counts,
                names="Method",
                values="Count",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Pastel1,
                template="plotly_dark"
            )
            fig_pay.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350)
            st.plotly_chart(fig_pay, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown("### 🍔 Best-Selling Menu Items")
        if "Product" in filtered_df.columns:
            top_products = filtered_df.groupby("Product")["Total_Sales"].sum().reset_index().sort_values("Total_Sales", ascending=False).head(6)
            
            fig_prod = px.bar(
                top_products,
                x="Product",
                y="Total_Sales",
                color="Total_Sales",
                color_continuous_scale=px.colors.sequential.Electric,
                template="plotly_dark",
                labels={"Total_Sales": "Revenue ($)", "Product": "Menu Product"}
            )
            fig_prod.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350, coloraxis_showscale=False)
            st.plotly_chart(fig_prod, use_container_width=True)

    with row2_col2:
        st.markdown("### 🪑 Order Type Mix (Dine-In vs Takeaway)")
        if "Purchase Type" in filtered_df.columns:
            type_counts = filtered_df["Purchase Type"].value_counts().reset_index()
            type_counts.columns = ["Type", "Count"]
            
            fig_type = px.pie(
                type_counts,
                names="Type",
                values="Count",
                hole=0.45,
                color_discrete_sequence=["#f78166", "#3fb950"],
                template="plotly_dark"
            )
            fig_type.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350)
            st.plotly_chart(fig_type, use_container_width=True)

# --- TAB 2: PREDICTIVE ENGINE ---
with tab2:
    st.markdown("## 🔮 Machine Learning Scenario Simulator")
    st.markdown("Simulate order combinations below to forecast transaction revenue using a Random Forest model.")
    st.markdown("<br>", unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns([1, 1], gap="large")

    with sim_col1:
        st.markdown("### ⚙️ Simulation Parameters")
        
        input_price = st.slider("💵 Unit Price ($)", float(df["Price"].min()), float(df["Price"].max()), float(df["Price"].median()))
        input_quantity = st.slider("📦 Order Quantity", int(df["Quantity"].min()), int(df["Quantity"].max()), 3)
        input_city = st.selectbox("📍 Target Branch", df["City"].unique().tolist())
        input_type = st.selectbox("🍽️ Fulfillment Channel", df["Purchase Type"].unique().tolist() if "Purchase Type" in df.columns else ["Dine-In"])
        input_payment = st.selectbox("💳 Payment Channel", df["Payment Method"].unique().tolist() if "Payment Method" in df.columns else ["Credit Card"])

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🚀 Forecast Transaction Value", use_container_width=True)

    with sim_col2:
        st.markdown("### 🎯 Predicted Outcome & Insights")
        
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

            st.markdown(f"""
                <div style="background: #161b22; border: 2px solid #238636; border-radius: 12px; padding: 25px; text-align: center; margin-bottom: 20px;">
                    <div style="font-size: 1rem; color: #8b949e; text-transform: uppercase; font-weight: 700;">Forecasted Order Value</div>
                    <div style="font-size: 3rem; color: #3fb950; font-weight: 800; margin: 10px 0;">${prediction:,.2f}</div>
                    <div style="font-size: 0.9rem; color: #8b949e;">⚡ AI Model Confidence ($R^2$ Score): <strong style="color: #f0f6fc;">{model_r2:.2f}</strong></div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 💡 Strategic Business Prescription:")
            
            if prediction > 80:
                st.success("🌟 **High-Value Order Tier:** Automatically enroll customer in VIP loyalty perk status and present high-margin dessert add-ons!")
            elif prediction > 40:
                st.info("⚡ **Standard Order Tier:** Strong opportunity to suggest combo deals or drink upgrades at point-of-sale.")
            else:
                st.warning("🔹 **Value Order Tier:** Recommend automated digital promotional vouchers for their next visit.")
        else:
            st.info("👈 Set your scenario parameters on the left and click **'Forecast Transaction Value'** to view AI predictions.")

# --- TAB 3: DATA EXPLORER ---
with tab3:
    st.markdown("## 📜 Transaction Logs Explorer")
    st.markdown("Filter, inspect, and export underlying order dataset records.")
    
    st.dataframe(filtered_df if 'filtered_df' in locals() else df, use_container_width=True, height=450)

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #8b949e; font-size: 0.85rem;'>Enterprise Intelligence Platform • Built by Manikar Raj</p>", unsafe_allow_html=True)