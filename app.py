import streamlit as st
import pandas as pd
import sqlite3
import os
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- PAGE SETUP (Premium UI) ---
st.set_page_config(page_title="Cafe Pro Dashboard", page_icon="☕", layout="wide")

# Custom CSS to make it look more like a modern web app
st.markdown("""
<style>
    .main {background-color: #f4f6f9;}
    h1 {color: #1E3D59;}
    h2, h3 {color: #2F5D62;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; font-size: 18px;}
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    db_file_path = os.path.join(current_directory, 'cafe_data.db')
    csv_file_path = os.path.join(current_directory, 'restaurant_orders.csv')
    
    # --- CLOUD DEPLOYMENT FIX ---
    # If the database doesn't exist (because GitHub ignored it), build it now!
    if not os.path.exists(db_file_path):
        try:
            raw_df = pd.read_csv(csv_file_path)
            setup_conn = sqlite3.connect(db_file_path)
            raw_df.to_sql('orders', setup_conn, if_exists='replace', index=False)
            setup_conn.close()
        except Exception as e:
            st.error(f"Could not build database on the cloud: {e}")
            return None
    # ----------------------------

    try:
        conn = sqlite3.connect(db_file_path)
        df = pd.read_sql("SELECT * FROM orders;", conn)
        conn.close()
        
        # Clean Data immediately
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
        df['Payment Method'] = df['Payment Method'].str.strip()
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Could not load database: {e}")
        return None

df = load_data()

if df is not None:
    
    # --- SIDEBAR (Global Filters) ---
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3075/3075977.png", width=100)
    st.sidebar.title("Dashboard Controls")
    st.sidebar.write("Filter the data below:")
    
    city_list = df['City'].unique().tolist()
    selected_cities = st.sidebar.multiselect("Select Cities", city_list, default=city_list)
    
    # Apply Filter
    filtered_df = df[df['City'].isin(selected_cities)]

    # --- MAIN HEADER ---
    st.title("☕ Cafe Operations & AI Dashboard")
    st.markdown("Welcome to the **Premium Analytics Workspace**. Explore interactive insights and leverage predictive machine learning.")

    # --- TAB LAYOUT ---
    tab1, tab2 = st.tabs(["📈 Advanced Analytics", "🤖 AI Predictor"])

    # ==========================================
    # TAB 1: ADVANCED ANALYTICS (PLOTLY)
    # ==========================================
    with tab1:
        # Top Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Orders", len(filtered_df))
        col2.metric("Active Cities", filtered_df['City'].nunique())
        col3.metric("Top Product", filtered_df['Product'].mode()[0] if not filtered_df.empty else "N/A")
        col4.metric("Top Payment", filtered_df['Payment Method'].mode()[0] if not filtered_df.empty else "N/A")
        
        st.divider()

        if not filtered_df.empty:
            # ROW 1: Timeline Scatter Plot & Donut Chart
            row1_col1, row1_col2 = st.columns([2, 1]) 
            
            with row1_col1:
                st.subheader("Daily Order Volume (Time Series)")
                # Group by date to get order counts
                time_data = filtered_df.groupby('Date').size().reset_index(name='Orders')
                # Create an awesome interactive Scatter/Line plot
                fig_scatter = px.line(time_data, x='Date', y='Orders', markers=True, 
                                      title="Orders over Time", 
                                      line_shape="spline",
                                      color_discrete_sequence=['#ff7f0e'])
                fig_scatter.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=2, color='DarkSlateGrey')))
                st.plotly_chart(fig_scatter, use_container_width=True)

            with row1_col2:
                st.subheader("Payment Preferences")
                pay_data = filtered_df['Payment Method'].value_counts().reset_index()
                pay_data.columns = ['Method', 'Count']
                # Create a Donut chart
                fig_donut = px.pie(pay_data, names='Method', values='Count', hole=0.5, 
                                   color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_donut.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_donut, use_container_width=True)

            # ROW 2: Advanced Treemap (Hierarchy)
            st.subheader("City & Product Breakdown (Treemap)")
            st.write("Click on a City box to zoom in and see the specific products sold there!")
            tree_data = filtered_df.groupby(['City', 'Product']).size().reset_index(name='Count')
            fig_tree = px.treemap(tree_data, path=[px.Constant("All Locations"), 'City', 'Product'], 
                                  values='Count', color='Count', color_continuous_scale='Blues')
            st.plotly_chart(fig_tree, use_container_width=True)
        else:
            st.warning("No data available for the selected cities. Please change your filter.")

    # ==========================================
    # TAB 2: MACHINE LEARNING PREDICTOR
    # ==========================================
    with tab2:
        st.header("🧠 Smart Checkout AI")
        st.write("Input the customer's details below. The AI will predict their payment method and show you its **confidence level**.")

        # Train model logic
        ml_df = df.copy()
        ml_df = ml_df[ml_df['Payment Method'] != 'Gift Card'] # Drop Gift Cards
        ml_df['DayOfWeek'] = ml_df['Date'].dt.dayofweek
        ml_df['Is_Weekend'] = ml_df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

        X = ml_df[['Product', 'City', 'Is_Weekend']]
        y = ml_df['Payment Method']
        X_encoded = pd.get_dummies(X, columns=['Product', 'City'])
        training_columns = X_encoded.columns

        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))

        st.info(f"Model trained on {len(ml_df)} records. Current Accuracy: **{accuracy * 100:.1f}%**")
        st.divider()

        # UI Layout for Inputs
        input_col1, input_col2, input_col3 = st.columns(3)
        with input_col1:
            selected_product = st.selectbox("🛒 Select Product:", ml_df['Product'].unique())
        with input_col2:
            selected_city = st.selectbox("🏙️ Select City:", ml_df['City'].unique())
        with input_col3:
            is_weekend = st.selectbox("📅 Is it the weekend?", ["No (Weekday)", "Yes (Weekend)"])
            weekend_val = 1 if is_weekend == "Yes (Weekend)" else 0

        # Prediction Button
        if st.button("🔮 Predict Payment Method", use_container_width=True, type="primary"):
            
            with st.spinner("AI is calculating..."):
                user_input = pd.DataFrame({'Product': [selected_product], 'City': [selected_city], 'Is_Weekend': [weekend_val]})
                user_encoded = pd.get_dummies(user_input)
                user_encoded = user_encoded.reindex(columns=training_columns, fill_value=0)
                
                prediction = model.predict(user_encoded)[0]
                probabilities = model.predict_proba(user_encoded)[0]
                classes = model.classes_

                st.success(f"### 🤖 Prediction: The customer will likely use **{prediction}**")

                # Build a probability chart!
                st.subheader("AI Confidence Breakdown")
                prob_df = pd.DataFrame({'Payment Method': classes, 'Confidence': probabilities})
                fig_prob = px.bar(prob_df, x='Confidence', y='Payment Method', orientation='h',
                                  text_auto='.1%', color='Payment Method', 
                                  color_discrete_sequence=['#2ecc71', '#3498db'])
                fig_prob.update_layout(xaxis_tickformat='.0%', showlegend=False)
                st.plotly_chart(fig_prob, use_container_width=True)