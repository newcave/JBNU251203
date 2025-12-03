%%writefile app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# ===============================================
# 1. Synthetic chlorine decay simulation function
# ===============================================
def chlorine_decay(temp, dose, t):
    k = 0.015 + 0.002 * (temp - 10)
    residual = dose * np.exp(-k * t)
    noise = np.random.normal(0, 0.05)
    return max(residual + noise, 0)


# ===============================================
# 2. Generate synthetic dataset
# ===============================================
np.random.seed(42)
N = 3000

temps = np.random.uniform(5, 30, N)
doses = np.random.uniform(0.5, 3.0, N)
times = np.random.uniform(0, 6, N)

residuals = np.array([chlorine_decay(temps[i], doses[i], times[i]) for i in range(N)])

df = pd.DataFrame({
    "Temperature": temps,
    "Input_Dose": doses,
    "Time_hr": times,
    "Residual_Chlorine": residuals
})

# ===============================================
# 3. Train Decision Tree model
# ===============================================
X = df[["Temperature", "Input_Dose", "Time_hr"]]
y = df["Residual_Chlorine"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeRegressor(max_depth=6)
model.fit(X_train, y_train)


# ===============================================
# 4. Streamlit UI
# ===============================================
st.title("💧 Residual Chlorine Prediction (6-hour process)")
st.write("Decision Tree 기반 잔류염소 예측 (수온 + 투입염소 + 시간)")

st.sidebar.header("입력값 선택")

temp = st.sidebar.slider("수온 (°C)", 5.0, 30.0, 20.0)
dose = st.sidebar.slider("투입 염소 (mg/L)", 0.5, 3.0, 1.5)
hour = st.sidebar.slider("시간 (hr)", 0.0, 6.0, 0.0)


# ===============================================
# 5. Real-time prediction
# ===============================================
def realtime_predict(temp, dose, hour):
    pred = model.predict(pd.DataFrame({
        "Temperature": [temp],
        "Input_Dose": [dose],
        "Time_hr": [hour]
    }))[0]
    return max(pred, 0)


predicted_value = realtime_predict(temp, dose, hour)

st.subheader("⏱ 실시간 예측 결과")
st.metric(label="예측 잔류염소 (mg/L)", value=f"{predicted_value:.3f}")


# ===============================================
# 6. Plot full decay curve
# ===============================================
t_range = np.linspace(0, 6, 50)

true_curve = [chlorine_decay(temp, dose, t) for t in t_range]
pred_curve = model.predict(pd.DataFrame({
    "Temperature": [temp]*50,
    "Input_Dose": [dose]*50,
    "Time_hr": t_range
}))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_range, true_curve, label="실제(시뮬레이션)", linewidth=3)
ax.plot(t_range, pred_curve, label="ML 예측", linestyle="--", linewidth=3)
ax.set_xlabel("시간 (hr)")
ax.set_ylabel("잔류염소 (mg/L)")
ax.set_title("잔류염소 Decay 곡선")
ax.grid(True)
ax.legend()

st.pyplot(fig)

st.write("---")
st.write("교육용 시뮬레이션 모델 (Decision Tree 기반)")
