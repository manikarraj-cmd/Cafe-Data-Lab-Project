import pandas as pd
import sqlite3
import os

# Get the current folder path
current_directory = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(current_directory, 'restaurant_orders.csv')
db_file_path = os.path.join(current_directory, 'cafe_data.db')

try:
    print("Step 1: Reading CSV file...")
    df = pd.read_csv(csv_file_path)
    
    print("Step 2: Connecting to SQLite database...")
    conn = sqlite3.connect(db_file_path)
    
    print("Step 3: Transferring data...")
    df.to_sql('orders', conn, if_exists='replace', index=False)
    
    print("\n✅ SUCCESS! Data loaded into 'cafe_data.db'")
    
    # Preview first 3 rows
    test_results = pd.read_sql("SELECT * FROM orders LIMIT 3;", conn)
    print("\nPreview:")
    print(test_results)

except FileNotFoundError:
    print("\n❌ ERROR: Could not find 'restaurant_orders.csv'. Make sure it is in the same folder!")
except Exception as e:
    print(f"\n❌ An error occurred: {e}")
finally:
    if 'conn' in locals():
        conn.close()