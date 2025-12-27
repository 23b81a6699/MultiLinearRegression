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
    page_title="Multiple Linear Regression",
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
    <h1>Multiple Linear Regression</h1>
    <p>Predict <b>Sales</b> from <b>TV, Radio & Newspaper</b> advertising spend</p>
</div>
""", unsafe_allow_html=True)

# =========================
# Load Dataset
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("advertising.csv")

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
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

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
st.subheader("TV Advertising vs Sales")

fig, ax = plt.subplots()
ax.scatter(df["TV"], df["Sales"], alpha=0.6)

tv_vals = np.linspace(df["TV"].min(), df["TV"].max(), 100)
avg_radio = df["Radio"].mean()
avg_news = df["Newspaper"].mean()

X_line = pd.DataFrame({
    "TV": tv_vals,
    "Radio": avg_radio,
    "Newspaper": avg_news
})

X_line_scaled = scaler.transform(X_line)
y_line = model.predict(X_line_scaled)

ax.plot(tv_vals, y_line, color="red")
ax.set_xlabel("TV Advertising")
ax.set_ylabel("Sales")

st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Performance Section
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")

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
        <b>TV:</b> {model.coef_[0]:.3f}<br>
        <b>Radio:</b> {model.coef_[1]:.3f}<br>
        <b>Newspaper:</b> {model.coef_[2]:.3f}<br><br>
        <b>Intercept:</b> {model.intercept_:.3f}
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# Prediction Section
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Sales")

tv = st.slider(
    "TV Advertising Budget",
    float(df.TV.min()),
    float(df.TV.max()),
    150.0
)

radio = st.slider(
    "Radio Advertising Budget",
    float(df.Radio.min()),
    float(df.Radio.max()),
    20.0
)

news = st.slider(
    "Newspaper Advertising Budget",
    float(df.Newspaper.min()),
    float(df.Newspaper.max()),
    30.0
)

input_scaled = scaler.transform([[tv, radio, news]])
pred_sales = model.predict(input_scaled)[0]

st.markdown(
    f'<div class="prediction-box">Predicted Sales: {pred_sales:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)