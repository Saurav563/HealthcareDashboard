import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load prediction data
df = pd.read_csv("final_predictions.csv")
df['Date'] = pd.to_datetime(df['Date'])


# Optional: load ARIMA forecast
forecast_arima = pd.Series(
    [77.88, 78.33, 77.79, 78.32, 77.81, 78.31, 77.82],
    index=pd.date_range(start='2025-01-01', periods=7, freq='D'),
    name="Predicted_Patient_Volume"
)

# Optional: load staff optimization result
staff_allocation = np.array([4, 6, 7, 5, 8])
staff_names = [f"Staff {i+1}" for i in range(len(staff_allocation))]

# Optional: simulate overcrowding classifier output
predicted_risk = 1  # 0 = Normal, 1 = High Risk


# Page setup
st.set_page_config(page_title="Hospital ML Dashboard", layout="wide")
st.title("🏥 Hospital Forecasting & Optimization Dashboard")

# Sidebar filters
st.sidebar.title("Filters")
date_range = st.sidebar.date_input("Select Date Range", [])
if date_range:
    df = df[(df['Date'] >= pd.to_datetime(date_range[0])) & (df['Date'] <= pd.to_datetime(date_range[1]))]

# Forecast plotting function
def plot_forecast(actual_col, predicted_col, title):
    fig, ax = plt.subplots()
    ax.plot(df['Date'], df[actual_col], label='Actual', marker='o')
    ax.plot(df['Date'], df[predicted_col], label='Predicted', marker='x')
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(actual_col.replace("_", " "))
    ax.legend()
    st.pyplot(fig)

# Section 1: Regression Models

st.header("📈 Forecasted Metrics")
plot_forecast("Actual_Staff", "Predicted_Staff", "Staff Forecast")
plot_forecast("Actual_ICU", "Predicted_ICU", "ICU Admissions Forecast")
plot_forecast("Actual_Ventilator", "Predicted_Ventilator", "Ventilator Usage Forecast")
plot_forecast("Actual_LOS", "Predicted_LOS", "Length of Stay Forecast")

# Section 2: Overcrowding Risk Classifier
st.header("🚨 Overcrowding Risk")
risk_label = "🟢 Normal" if predicted_risk == 0 else "🔴 High Risk"
st.markdown(f"**Predicted Risk Level:** {risk_label}")

# Section 3: Time Series Forecast
st.header("📅 Next Week's Patient Volume Forecast")
st.line_chart(forecast_arima)

# Section 4: Staff Optimization
st.header("👥 Staff Allocation Plan")
st.bar_chart(pd.Series(staff_allocation, index=staff_names))
st.metric("Total Assigned Hours", int(sum(staff_allocation)))

# Footer
st.markdown("---")
st.caption("Built by Saurav | AI-Powered Hospital Planning Suite")

