import pandas as pd
import sqlite3
import os
import matplotlib.pyplot as plt

current_directory = os.path.dirname(os.path.abspath(__file__))
db_file_path = os.path.join(current_directory, 'cafe_data.db')

try:
    conn = sqlite3.connect(db_file_path)
    
    print("--- 1. Total Records in Database ---")
    count_query = "SELECT COUNT(*) AS total_rows FROM orders;"
    count_result = pd.read_sql(count_query, conn)
    print(count_result)
    print("\n")

    print("--- 2. Most Popular Products ---")
    popular_query = """
    SELECT Product, COUNT(*) as Total_Orders 
    FROM orders 
    GROUP BY Product 
    ORDER BY Total_Orders DESC 
    LIMIT 5;
    """
    popular_result = pd.read_sql(popular_query, conn)
    print(popular_result)
    print("\n")
    
    # --- Visualization 1: Top 5 Products ---
    plt.figure(figsize=(10, 6))
    plt.bar(popular_result['Product'], popular_result['Total_Orders'], color='skyblue')
    plt.title('Top 5 Most Popular Products')
    plt.xlabel('Product')
    plt.ylabel('Total Orders')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    print("--- 3. Orders by City ---")
    city_query = """
    SELECT City, COUNT(*) as Total_Orders 
    FROM orders 
    GROUP BY City 
    ORDER BY Total_Orders DESC;
    """
    city_result = pd.read_sql(city_query, conn)
    print(city_result)
    print("\n")
    
    # --- Visualization 2: Orders by City ---
    plt.figure(figsize=(10, 6))
    plt.bar(city_result['City'], city_result['Total_Orders'], color='lightgreen')
    plt.title('Total Orders by City')
    plt.xlabel('City')
    plt.ylabel('Total Orders')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    print("--- 4. Preferred Payment Methods ---")
    # Note: Because 'Payment Method' has a space in the name, SQL requires brackets [] around it
    payment_query = """
    SELECT [Payment Method], COUNT(*) as Total_Uses 
    FROM orders 
    GROUP BY [Payment Method] 
    ORDER BY Total_Uses DESC;
    """
    payment_result = pd.read_sql(payment_query, conn)
    print(payment_result)

    # --- Visualization 3: Payment Methods ---
    plt.figure(figsize=(8, 5))
    plt.bar(payment_result['Payment Method'], payment_result['Total_Uses'], color='salmon')
    plt.title('Preferred Payment Methods')
    plt.xlabel('Payment Method')
    plt.ylabel('Total Uses')
    plt.tight_layout()
    plt.show()

except sqlite3.OperationalError:
    print("❌ ERROR: Could not connect to the database. Did you run database_setup.py first?")
except Exception as e:
    print(f"❌ An error occurred: {e}")
finally:
    if 'conn' in locals():
        conn.close()