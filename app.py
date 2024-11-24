import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Title and Description
st.title("Climate Change Impact Analysis")
st.markdown("""
This app analyzes historical climate data, focusing on temperature and rainfall trends, 
seasonal patterns, anomaly detection, and predictive modeling.
""")

# File Upload
uploaded_file = st.file_uploader("Upload Climate Data CSV File", type=["csv"])
if uploaded_file:
    # Load and preprocess data
    df = pd.read_csv(uploaded_file, skiprows=11)

    # Rename columns
    df.columns = ['Parameter', 'Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']
    df = df[1:]
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    months_and_annual = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']
    df[months_and_annual] = df[months_and_annual].apply(pd.to_numeric, errors='coerce')

    # Separate data into temperature and rainfall
    df_temperature = df[df['Parameter'] == 'TS'].copy()
    df_rainfall = df[df['Parameter'] == 'PRECTOTCORR_SUM'].copy()

    # Sidebar navigation
    analysis_options = st.sidebar.selectbox(
        "Choose Analysis",
        ["Annual Trends", "Seasonal Decomposition", "Anomaly Detection", "Modeling & Forecasting"]
    )

    # --- Annual Trends ---
    if analysis_options == "Annual Trends":
        st.header("Annual Trends: Temperature and Rainfall")

        # Plot temperature and rainfall trends
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        axes[0].plot(df_temperature['Year'], df_temperature['Annual'], color='orange', label='Temperature (°C)')
        axes[0].set_title('Annual Temperature Trend (1981 - 2022)')
        axes[0].set_ylabel('Average Temperature (°C)')
        axes[0].legend()

        axes[1].plot(df_rainfall['Year'], df_rainfall['Annual'], color='blue', label='Rainfall (mm)')
        axes[1].set_title('Annual Rainfall Trend (1981 - 2022)')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Total Annual Rainfall (mm)')
        axes[1].legend()

        st.pyplot(fig)

    # --- Seasonal Decomposition ---
    elif analysis_options == "Seasonal Decomposition":
        st.header("Seasonal Decomposition: Temperature and Rainfall")
        month = st.selectbox("Choose Month for Seasonal Decomposition", months_and_annual[:12])

        temp_series = df_temperature[month]
        rain_series = df_rainfall[month]

        # Decompose temperature
        st.subheader("Temperature Decomposition")
        decompose_temp = seasonal_decompose(temp_series, period=12, model='additive', extrapolate_trend='freq')
        fig_temp = decompose_temp.plot()
        st.pyplot(fig_temp)

        # Decompose rainfall
        st.subheader("Rainfall Decomposition")
        decompose_rain = seasonal_decompose(rain_series, period=12, model='additive', extrapolate_trend='freq')
        fig_rain = decompose_rain.plot()
        st.pyplot(fig_rain)

    # --- Anomaly Detection ---
    elif analysis_options == "Anomaly Detection":
        st.header("Anomaly Detection")

        # Detect anomalies for temperature
        temp_mean, temp_std = df_temperature['Annual'].mean(), df_temperature['Annual'].std()
        temp_anomalies = df_temperature[
            (df_temperature['Annual'] > temp_mean + 2 * temp_std) |
            (df_temperature['Annual'] < temp_mean - 2 * temp_std)
        ]

        # Detect anomalies for rainfall
        rain_mean, rain_std = df_rainfall['Annual'].mean(), df_rainfall['Annual'].std()
        rain_anomalies = df_rainfall[
            (df_rainfall['Annual'] > rain_mean + 2 * rain_std) |
            (df_rainfall['Annual'] < rain_mean - 2 * rain_std)
        ]

        st.subheader("Temperature Anomalies")
        st.write(temp_anomalies[['Year', 'Annual']])

        st.subheader("Rainfall Anomalies")
        st.write(rain_anomalies[['Year', 'Annual']])

    # --- Modeling and Forecasting ---
    elif analysis_options == "Modeling & Forecasting":
        st.header("Modeling and Forecasting")

        # Polynomial Regression
        X = df_temperature['Annual'].values.reshape(-1, 1)
        y = df_rainfall['Annual'].values

        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        st.subheader("Polynomial Regression")
        st.write(f"Mean Squared Error: {mse:.2f}")

        # ARIMA Forecasting
        temperature_series = df_temperature.set_index(pd.to_datetime(df_temperature['Year'], format='%Y'))['Annual']
        arima_model = ARIMA(temperature_series, order=(1, 1, 1))
        arima_result = arima_model.fit()

        forecast = arima_result.get_forecast(steps=5)
        forecast_df = forecast.summary_frame()

        st.subheader("ARIMA Forecast")
        st.write(forecast_df)

        # Plot forecast
        st.line_chart(forecast_df['mean'])

else:
    st.warning("Please upload a CSV file to proceed.")
