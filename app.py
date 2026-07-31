import os
import pandas as pd
import streamlit as st

# --- 1. ROBUST DATA LOADING ---
@st.cache_data
def load_data():
    # Resolve absolute path relative to this script file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "restaurant_orders.csv")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        # Fallback to local execution directory if running directly
        df = pd.read_csv("restaurant_orders.csv")

    # Clean text columns
    if "City" in df.columns:
        df["City"] = df["City"].astype(str).str.strip()

    return df


df = load_data()

# --- 2. SIDEBAR CITY FILTER WITH DEFAULTS ---
st.sidebar.header("Filter Options")

# Extract unique cities from data
available_cities = sorted(df["City"].unique().tolist())

# SET ALL CITIES SELECTED BY DEFAULT SO IT DOESN'T START EMPTY
selected_cities = st.sidebar.multiselect(
    "Select City:", options=available_cities, default=available_cities
)

# --- 3. FILTERING LOGIC ---
if not selected_cities:
    st.warning("⚠️ Please select at least one city from the sidebar.")
else:
    filtered_df = df[df["City"].isin(selected_cities)]

    if filtered_df.empty:
        st.error("No data found for the selected cities.")
    else:
        st.success(
            f"Displaying data for **{len(filtered_df):,}** orders across selected cities."
        )

        # Render your metrics and charts below using filtered_df
        st.dataframe(filtered_df)