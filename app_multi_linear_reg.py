import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Car Price Prediction",
    layout="centered"
)

# =========================
# Load CSS
# =========================
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")

# =========================
# Title Section
# =========================
st.markdown("""
<div class="card">
    <h1>Car Price Prediction</h1>
    <p>Predict <b>Car Price</b> using <b>Engine Size, Horsepower & Mileage</b></p>
</div>
""", unsafe_allow_html=True)

# =========================
# Load Dataset
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("CarPrice_Assignment.csv")

df = load_data()

# =========================
# Dataset Preview
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Prepare Data
# =========================
features = ["enginesize", "horsepower", "citympg"]
target = "price"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# Train Model
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# =========================
# Model Metrics
# =========================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X.shape[1] - 1)

# =========================
# Visualization
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Engine Size vs Car Price")

fig, ax = plt.subplots()
ax.scatter(df["enginesize"], df["price"], alpha=0.6)

engine_vals = np.linspace(df["enginesize"].min(), df["enginesize"].max(), 100)
avg_hp = df["horsepower"].mean()
avg_mpg = df["citympg"].mean()

X_line = pd.DataFrame({
    "enginesize": engine_vals,
    "horsepower": avg_hp,
    "citympg": avg_mpg
})

X_line_scaled = scaler.transform(X_line)
y_line = model.predict(X_line_scaled)

ax.plot(engine_vals, y_line, color="red")
ax.set_xlabel("Engine Size")
ax.set_ylabel("Car Price")

st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Performance Section
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:,.2f}")
c2.metric("RMSE", f"{rmse:,.2f}")

c3, c4 = st.columns(2)
c3.metric("R²", f"{r2:.3f}")
c4.metric("Adjusted R²", f"{adj_r2:.3f}")

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Model Coefficients
# =========================
st.markdown(f"""
<div class="card">
    <h3>Model Coefficients</h3>
    <p>
        <b>Engine Size:</b> {model.coef_[0]:.2f}<br>
        <b>Horsepower:</b> {model.coef_[1]:.2f}<br>
        <b>City MPG:</b> {model.coef_[2]:.2f}<br><br>
        <b>Intercept:</b> {model.intercept_:,.2f}
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# Prediction Section
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Car Price")

engine = st.slider(
    "Engine Size",
    float(df.enginesize.min()),
    float(df.enginesize.max()),
    float(df.enginesize.mean())
)

horsepower = st.slider(
    "Horsepower",
    float(df.horsepower.min()),
    float(df.horsepower.max()),
    float(df.horsepower.mean())
)

citympg = st.slider(
    "City Mileage (MPG)",
    float(df.citympg.min()),
    float(df.citympg.max()),
    float(df.citympg.mean())
)

input_scaled = scaler.transform([[engine, horsepower, citympg]])
pred_price = model.predict(input_scaled)[0]

st.markdown(
    f'<div class="prediction-box">Predicted Car Price: ₹ {pred_price:,.0f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)