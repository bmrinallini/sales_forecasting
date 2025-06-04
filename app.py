import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Rossmann Sales Forecast", layout="centered")
st.title("📊 Rossmann Store Sales Forecasting App")

@st.cache_data
def load_data():
    train = pd.read_csv("data/train.csv")
    store = pd.read_csv("data/store.csv")
    train = pd.merge(train, store, how="left", on="Store")
    train = train[(train["Open"] == 1) & (train["Sales"] > 0)]
    train["Date"] = pd.to_datetime(train["Date"])
    return train

train = load_data()

# Feature Engineering
train["Day"] = train["Date"].dt.day
train["Month"] = train["Date"].dt.month
train["Year"] = train["Date"].dt.year

# Add new time features
train["DayOfWeek"] = train["Date"].dt.dayofweek
train["WeekOfYear"] = train["Date"].dt.isocalendar().week.astype(int)

# One-hot encode StateHoliday
train["StateHoliday"] = train["StateHoliday"].astype(str)
train = pd.get_dummies(train, columns=["StateHoliday"], drop_first=True)

store_id = st.selectbox("🏬 Select Store ID", sorted(train["Store"].unique()))
store_data = train[train["Store"] == store_id].copy()

# Sort by date for lag features
store_data = store_data.sort_values("Date")

# Lag and rolling features
store_data["Sales_lag_7"] = store_data["Sales"].shift(7)
store_data["Sales_roll_7"] = store_data["Sales"].rolling(window=7).mean()
store_data["Sales_roll_30"] = store_data["Sales"].rolling(window=30).mean()

# Define features list
features = ["Day", "Month", "Year", "Promo", "SchoolHoliday", "DayOfWeek", "WeekOfYear"] + \
           [col for col in store_data.columns if "StateHoliday_" in col] + \
           ["Sales_lag_7", "Sales_roll_7", "Sales_roll_30"]

# Prepare X and y, fill NaNs with 0 for lag/rolling features
X = store_data[features].fillna(0)
y = store_data["Sales"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Train model with XGBoost
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    objective='reg:squarederror',
    tree_method='hist'  # fast training on large datasets
)
model.fit(X_train, y_train)

# Future prediction dates
future_days = 7
last_date = store_data["Date"].max()
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, future_days + 1)]

# Prepare future data
future_df = pd.DataFrame({
    "Date": future_dates,
    "Day": [d.day for d in future_dates],
    "Month": [d.month for d in future_dates],
    "Year": [d.year for d in future_dates],
    "Promo": 0,
    "SchoolHoliday": 0,
    "DayOfWeek": [d.dayofweek for d in future_dates],
    "WeekOfYear": [d.isocalendar().week for d in future_dates],
})

# Add missing dummy columns for StateHoliday dummies
for col in [c for c in train.columns if "StateHoliday_" in c]:
    future_df[col] = 0

# Lag and rolling features for future data (fill with 0, no past sales info)
future_df["Sales_lag_7"] = 0
future_df["Sales_roll_7"] = 0
future_df["Sales_roll_30"] = 0

future_X = future_df[features]
future_preds = model.predict(future_X)

# 📈 Actual vs Predicted Sales (Last 30 Days)
st.subheader("📈 Actual vs Predicted Sales (Last 30 Days)")

# Prepare test data
y_test = y_test.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_pred = pd.Series(model.predict(X_test))

# Select last 30 values
actual_sales = y_test[-30:].values
predicted_sales = y_pred[-30:].values
dates = store_data["Date"].iloc[-len(y_test):].reset_index(drop=True)[-30:]

# Smoothed predicted line
smoothed_preds = pd.Series(predicted_sales).rolling(window=3, center=True).mean()

spacing = 3
bar_width = 2
bar_positions = np.arange(len(actual_sales)) * spacing
line_positions = bar_positions + bar_width / 2  # center alignment

fig, ax = plt.subplots(figsize=(12, 6))

# Bars
ax.bar(bar_positions, actual_sales, width=bar_width, label="Actual Sales",
       color="#D6EAF8", edgecolor="black", linewidth=0.7)

# Smoothed line
ax.plot(line_positions, smoothed_preds, label="Predicted Sales (Smoothed)",
        color="orange", linestyle="--", linewidth=2, marker='o')

# Error lines
for x, actual, pred in zip(line_positions, actual_sales, predicted_sales):
    ax.vlines(x, min(actual, pred), max(actual, pred),
              color="gray", alpha=0.4, linestyle=":")

# Formatting
ax.set_title(f"Actual vs Predicted Sales - Store {store_id}", fontsize=16)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Sales", fontsize=12)
ax.set_xticks(bar_positions[::5])
ax.set_xticklabels([date.strftime("%b %d") for date in dates[::5]])

legend = ax.legend(loc="upper left", fontsize=12, frameon=True)
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(1.5)

st.pyplot(fig)

# 📅 Future Sales Forecast Table
st.subheader("📅 Future Sales Forecast (Next 7 Days)")
forecast_table = pd.DataFrame({
    "Date": future_dates,
    "Predicted Sales": future_preds.astype(int)
})
forecast_table.index = [f"Future Day {i+1} Sales" for i in range(future_days)]
st.write(forecast_table)
