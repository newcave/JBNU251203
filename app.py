import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# ===============================================
# 1. Chlorine decay simulation (for education)
# ===============================================
def chlorine_decay(temp, dose, t):
    k = 0.015 + 0.002 * (temp - 10)  # temperature-dependent decay rate
    residual = dose * np.exp(-k * t)
    noise = np.random.normal(0, 0.05)  # sensor noise
    return max(residual + noise, 0)

# ===============================================
# 2. Generate synthetic dataset
# ===============================================
@st.cache_data
def generate_dataset():
    np.random.seed(42)
    N = 3000

    temps = np.random.uniform(5, 30, N)
    doses = np.random.uniform(0.5, 3.0, N)
    times = np.random.uniform(0, 6, N)

    residuals = np.array([
        chlorine_decay(temps[i], doses[i], times[i]) for i in range(N)
    ])

    df = pd.DataFrame({
        "Temperature": temps,
        "Input_Dose": doses,
        "Time_hr": times,
        "Residual_Chlorine": residuals
    })
    return df

df = generate_dataset()

# ===============================================
# 3. Train Decision Tree model
# ===============================================
@st.cache_resource
def train_model():
    X = df[["Temperature", "Input_Dose", "Time_hr"]]
    y = df["Residual_Chlorine"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeRegressor(max_depth=6)
    model.fit(X_train, y_train)
    return model

model = train_model()

# ===============================================
# Streamlit App UI
# ===============================================
st.title("💧 Residual Chlorine Prediction (6-hour Process)")
st.write("This app demonstrates a **Decision Tree ML model** predicting residual chlorine\n"
         "based on **Temperature, Input Chlorine Dose, and Reaction Time (0–6 hours)**.")

st.sidebar.header("Input Parameters")

temp = st.sidebar.slider("Water Temperature (°C)", 5.0, 30.0, 20.0)
dose = st.sidebar.slider("Input Chlorine (mg/L)", 0.5, 3.0, 1.5)
hour = st.sidebar.slider("Reaction Time (hours)", 0.0, 6.0, 0.0)

# ===============================================
# Real-time prediction
# ===============================================
def realtime_predict(temp, dose, hour):
    pred = model.predict(pd.DataFrame({
        "Temperature": [temp],
        "Input_Dose": [dose],
        "Time_hr": [hour]
    }))[0]
    return max(pred, 0)

result = realtime_predict(temp, dose, hour)

st.subheader("⏱ Real-Time Predicted Residual Chlorine")
st.metric(label="Residual Chlorine (mg/L)", value=f"{result:.3f}")

# ===============================================
# Plot decay curve
# ===============================================
t_range = np.linspace(0, 6, 50)

true_curve = [chlorine_decay(temp, dose, t) for t in t_range]
pred_curve = model.predict(pd.DataFrame({
    "Temperature": [temp]*50,
    "Input_Dose": [dose]*50,
    "Time_hr": t_range
}))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_range, true_curve, label="True (Simulated)", linewidth=3)
ax.plot(t_range, pred_curve, label="ML Prediction", linestyle="--", linewidth=3)
ax.set_xlabel("Reaction Time (hr)")
ax.set_ylabel("Residual Chlorine (mg/L)")
ax.set_title("Residual Chlorine Decay Curve")
ax.grid(True)
ax.legend()

st.pyplot(fig)

st.write("---")
st.write("Developed for education on chlorine decay dynamics and ML prediction in water treatment.")

