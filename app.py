import streamlit as st
import pandas as pd
import os
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(page_title="Cafe Pro Dashboard", page_icon="☕", layout="wide")

# --- LOAD DATA (Direct CSV Load - Cloud Optimized) ---
@st.cache_data
def load_data():
    csv_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'restaurant_orders.csv')
    
    # We tell pandas that the first row is NOT a header
    column_names = ['Order ID', 'Date', 'Product', 'Order Amount', 'Total_Unknown', 'Payment Method', 'Customer Name', 'City']
    df = pd.read_csv(csv_file_path, names=column_names, header=None)
    
    # Now that we have named them, we can proceed with your existing code
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df['Payment Method'] = df['Payment Method'].str.strip()
    df = df.dropna()
    return df

df = load_data()

if df is not None:
    # --- SIDEBAR ---
    st.sidebar.title("Dashboard Controls")
    city_list = df['City'].unique().tolist()
    selected_cities = st.sidebar.multiselect("Select Cities", city_list, default=city_list)
    filtered_df = df[df['City'].isin(selected_cities)]

    # --- MAIN CONTENT ---
    st.title("☕ Cafe Operations & AI Dashboard")
    
    if filtered_df.empty:
        st.warning("No data found for the selected cities.")
    else:
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Orders", len(filtered_df))
        col2.metric("Active Cities", filtered_df['City'].nunique())
        col3.metric("Avg Order Value", f"₹{filtered_df['Order Amount'].mean():.2f}")

        # Charts
        tab1, tab2 = st.tabs(["📈 Advanced Analytics", "🤖 AI Predictor"])

        with tab1:
            st.subheader("Daily Order Volume")
            time_data = filtered_df.groupby('Date').size().reset_index(name='Orders')
            fig = px.line(time_data, x='Date', y='Orders', markers=True)
            st.plotly_chart(fig, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Payment Methods")
                pay_data = filtered_df['Payment Method'].value_counts().reset_index()
                pay_data.columns = ['Method', 'Count']
                fig2 = px.pie(pay_data, values='Count', names='Method', hole=0.4)
                st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.header("🧠 Smart Checkout AI")
            st.write("The model analyzes patterns to predict payment preferences.")
            # Mock UI for the AI Model
            if st.button("Run Prediction Sample"):
                st.success("AI Model Prediction: Customer likely to use Credit Card (Confidence: 88%)")
else:
    st.error("Could not load the dataset. Please ensure 'restaurant_orders.csv' is in the same folder as app.py.")