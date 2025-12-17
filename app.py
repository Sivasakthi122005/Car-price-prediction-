import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor  # CatBoost regression

# -------------------------------------------------------------
# 1️⃣ BACKGROUND IMAGE FUNCTION
# -------------------------------------------------------------
def set_background_image(image_path: str):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_path}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            min-height: 100vh;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------------------
# 2️⃣ SESSION STATE INITIALIZATION
# -------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "login"

def switch_page(page_name):
    st.session_state.page = page_name

# -------------------------------------------------------------
# 3️⃣ LOAD DATA
# -------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv("cardata.csv")
    df = df.drop(columns=['Unnamed: 0', 'Mileage Unit'], errors='ignore')
    df = df.dropna()
    return df

df = load_data()

# -------------------------------------------------------------
# 4️⃣ PREPARE DATA
# -------------------------------------------------------------
for col in ["name", "fuel", "seller_type", "transmission", "owner"]:
    df[col] = df[col].astype("category")

# Mapping categories
name_mapping = dict(enumerate(df["name"].cat.categories))
name_reverse_mapping = {v: k for k, v in name_mapping.items()}

fuel_mapping = dict(enumerate(df["fuel"].cat.categories))
fuel_reverse_mapping = {v: k for k, v in fuel_mapping.items()}

seller_mapping = dict(enumerate(df["seller_type"].cat.categories))
seller_reverse_mapping = {v: k for k, v in seller_mapping.items()}

trans_mapping = dict(enumerate(df["transmission"].cat.categories))
trans_reverse_mapping = {v: k for k, v in trans_mapping.items()}

owner_mapping = dict(enumerate(df["owner"].cat.categories))
owner_reverse_mapping = {v: k for k, v in owner_mapping.items()}

# Convert to numeric codes
for col in ["name", "fuel", "seller_type", "transmission", "owner"]:
    df[col] = df[col].cat.codes

X = df[["name", "year", "km_driven", "fuel", "seller_type", "transmission", "owner"]]
y = df["selling_price"]

# -------------------------------------------------------------
# 5️⃣ TRAIN OR LOAD CatBoost MODEL
# -------------------------------------------------------------
model_file = "car_price_model_catboost.pkl"

if os.path.exists(model_file):
    model = joblib.load(model_file)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        verbose=0,
        random_seed=42
    )
    model.fit(X_train, y_train, cat_features=[0,3,4,5,6])
    joblib.dump(model, model_file)

# -------------------------------------------------------------
# 6️⃣ BACKGROUND IMAGE
# -------------------------------------------------------------
set_background_image("car-with-white-background-idea-designing_593294-302.jpg")

# -------------------------------------------------------------
# 7️⃣ LOGIN PAGE
# -------------------------------------------------------------
if st.session_state.page == "login":
    st.title("🔐 Login to Car Price Predictor")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            switch_page("home")
        else:
            st.error("❌ Invalid Username or Password")

# -------------------------------------------------------------
# 8️⃣ HOME PAGE
# -------------------------------------------------------------
elif st.session_state.page == "home":
    st.title("🚗 Welcome to Car Price Predictor!")
    st.markdown("### Choose an option below 👇")

    if st.button("Start Prediction"):
        switch_page("prediction")

    if st.button("Logout"):
        switch_page("login")

# -------------------------------------------------------------
# 9️⃣ PREDICTION PAGE
# -------------------------------------------------------------
elif st.session_state.page == "prediction":
    st.title("🚘 Car Price Prediction")
    st.sidebar.header("Enter Car Details")

    car_name = st.sidebar.selectbox("Car Name", sorted(name_reverse_mapping.keys()))
    year = st.sidebar.number_input(
        "Car Year", 
        min_value=int(df["year"].min()), 
        max_value=int(df["year"].max()), 
        step=1
    )
    kms_driven = st.sidebar.number_input(
        "Kilometers Driven", 
        min_value=0, 
        max_value=int(df["km_driven"].max()), 
        step=500
    )
    fuel = st.sidebar.selectbox("Fuel Type", sorted(fuel_reverse_mapping.keys()))
    seller_type = st.sidebar.selectbox("Seller Type", sorted(seller_reverse_mapping.keys()))
    transmission = st.sidebar.selectbox("Transmission", sorted(trans_reverse_mapping.keys()))
    owner = st.sidebar.selectbox("Previous Owner Type", sorted(owner_reverse_mapping.keys()))

    if st.sidebar.button("Predict Price"):
        if year < 1990 or year > 2025:
            st.error("⚠️ Enter a valid year between 1990 and 2025.")
        else:
            input_data = pd.DataFrame([[ 
                name_reverse_mapping[car_name],
                year,
                kms_driven,
                fuel_reverse_mapping[fuel],
                seller_reverse_mapping[seller_type],
                trans_reverse_mapping[transmission],
                owner_reverse_mapping[owner],
            ]], columns=X.columns)

            # CatBoost prediction
            base_price = model.predict(input_data)[0]

            # 🔹 Apply depreciation based on previous owner
            owner_factor_dict = {
                "First Owner": 1.0,   # No reduction
                "Second Owner": 0.7,  # 30% reduction
                "Third Owner": 0.5,   # 50% reduction
                "Fourth & Above": 0.4 # 60% reduction
            }

            dep_factor = owner_factor_dict.get(owner, 1.0)
            adjusted_price = base_price * dep_factor

            # Simulate uncertainty (±5%) on adjusted price
            price_std = adjusted_price * 0.05
            daily_fluctuations = np.random.normal(0, price_std / adjusted_price, 30)
            future_prices = adjusted_price * (1 + daily_fluctuations)
            best_day = np.argmin(future_prices) + 1
            best_price = future_prices[best_day - 1]

            # ✅ Convert to lakhs or crores automatically
            def format_price(amount):
                if amount >= 1e7:
                    return round(amount / 1e7, 2), "Cr"
                else:
                    return round(amount / 1e5, 2), "Lakh"

            base_val, base_unit = format_price(adjusted_price)
            std_low, _ = format_price(adjusted_price - price_std)
            std_high, _ = format_price(adjusted_price + price_std)
            future_vals = [format_price(p)[0] for p in future_prices]
            best_val, best_unit = format_price(best_price)

            # Sidebar
            st.sidebar.success(f"💰 Estimated Price: ₹ {base_val} {base_unit}")
            st.sidebar.info(f"📉 Price Range (±5%): ₹ {std_low} - ₹ {std_high} {base_unit}")

            # Table and plot
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📅 Next 30-Day Price Predictions")
                st.table(pd.DataFrame({"Day": range(1, 31), f"Predicted Price (₹ {base_unit})": future_vals}))

            with col2:
                st.subheader("📈 Price Trend")
                plt.figure(figsize=(6, 4))
                plt.plot(range(1, 31), future_vals, marker='o', linestyle='-', color='b', alpha=0.7)
                plt.fill_between(range(1, 31),
                                 [v - (v*0.05) for v in future_vals],
                                 [v + (v*0.05) for v in future_vals],
                                 color='b', alpha=0.2, label="±5%")
                plt.axvline(best_day, color='r', linestyle='--', label=f'Best Day: {best_day}')
                plt.scatter(best_day, best_val, color='red', label=f'₹ {best_val} {best_unit}', zorder=3)
                plt.xlabel("Days")
                plt.ylabel(f"Price (₹ {base_unit})")
                plt.title("Car Price Trend Over 30 Days")
                plt.legend()
                st.pyplot(plt)

            st.success(f"✅ Best day to buy: **Day {best_day}** at ₹ {best_val} {best_unit}")

    if st.button("⬅️ Back to Home"):
        switch_page("home")
