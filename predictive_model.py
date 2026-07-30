import pandas as pd
import sqlite3
import os

# --- STEP 1: IMPORT MACHINE LEARNING LIBRARIES ---
# We use scikit-learn (sklearn), the industry standard for traditional ML
from sklearn.model_selection import train_test_split
# Replace RandomForest with LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

current_directory = os.path.dirname(os.path.abspath(__file__))
db_file_path = os.path.join(current_directory, 'cafe_data.db')

try:
    print("\n✅ Running the UPDATED model (No Gift Cards)...")
    # --- STEP 2: LOAD DATA FROM YOUR DATABASE ---
    print("1. Loading data from the database...")
    conn = sqlite3.connect(db_file_path)
    df = pd.read_sql("SELECT * FROM orders;", conn)
    conn.close()

    # --- STEP 3: FEATURE ENGINEERING (Data Prep) ---
    print("2. Preprocessing data (Feature Engineering)...")
    
    # 3A. Convert the 'Date' text into a real date object, then extract the Day of the Week (0=Monday, 6=Sunday)
    # Machine Learning models love numbers, so converting dates to day-numbers is highly effective!
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    # Drop any rows where data might be missing
    df = df.dropna()

    # --- CRITICAL FIX: CLEANING MESSY TEXT ---
    # Real-world data often has hidden spaces. We use .str.strip() to scrub them off!
    print("-> Scrubbing hidden spaces from text columns...")
    df['Payment Method'] = df['Payment Method'].str.strip()

    # --- FIX 1: DROP 'Gift Card' ROWS ---
    # Now that the text is clean, this filter will actually work!
    print("-> Applying Fix 1: Removing 'Gift Card' records...")
    df = df[df['Payment Method'] != 'Gift Card']
    
    # CRITICAL FIX: Reset the categories so 'Gift Card' is completely forgotten
    df['Payment Method'] = df['Payment Method'].cat.remove_unused_categories() if hasattr(df['Payment Method'], 'cat') else df['Payment Method']

    # --- FIX 2: ADVANCED FEATURE ENGINEERING ---
    # Let's create a new column: 'Is_Weekend'. If DayOfWeek is 5 (Sat) or 6 (Sun), it's a 1. Else 0.
    print("-> Applying Fix 2: Creating 'Is_Weekend' feature...")
    df['Is_Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

    # 3B. Define our Inputs (X) and our Target (y)
    # Notice we are now using our new 'Is_Weekend' feature instead of 'DayOfWeek'
    X = df[['Product', 'City', 'Is_Weekend']]
    y = df['Payment Method']

    # 3C. Encoding: Machine learning algorithms CANNOT read text like "Fries" or "London".
    # We use "One-Hot Encoding" (pd.get_dummies) to turn these text categories into 1s and 0s.
    X_encoded = pd.get_dummies(X, columns=['Product', 'City'])

    # --- STEP 4: TRAIN / TEST SPLIT ---
    # We hide 20% of the data (test_size=0.2) to test the model later, and train it on the other 80%.
    print("3. Splitting data into Training and Testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # --- STEP 5: TRAIN THE AI MODEL ---
    print("4. Training the Logistic Regression Machine Learning Model...")
    # Using Logistic Regression instead of Random Forest for this small dataset
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # --- STEP 6: TEST AND EVALUATE ---
    print("5. Testing the model on unseen data...\n")
    y_pred = model.predict(X_test)
    
    # Calculate how many it got right!
    accuracy = accuracy_score(y_test, y_pred)
    print("========================================")
    print(f"🎯 MODEL ACCURACY: {accuracy * 100:.2f}%")
    print("========================================\n")
    
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

except ImportError:
    print("❌ ERROR: scikit-learn is not installed. Please run: pip install scikit-learn")
except Exception as e:
    print(f"❌ An error occurred: {e}")