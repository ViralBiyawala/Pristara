import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Visualize the data
def visualize_data(electric_consumption_data):
    plt.figure(figsize=(12, 6))
    plt.plot(electric_consumption_data['Date'], electric_consumption_data['Value'], label='Energy Price Value', color='green')
    plt.title('Energy Price Value 2023-12-20 - 2023-12-20')
    plt.xlabel('Date')
    plt.ylabel('Consumption')
    plt.legend()
    plt.show()

# Visualize the forecasts
def visualize_forecasts(electric_consumption_data, regression_wavelet_forecast, hilbert_huang_forecast, arima_forecast, forecast_steps):
    plt.figure(figsize=(22, 14))
    plt.plot(np.append(electric_consumption_data[electric_consumption_data['Date'] >= electric_consumption_data['Date'].max() - datetime.timedelta(days=1)]['Date'], pd.date_range(start=electric_consumption_data['Date'].max(), periods=forecast_steps, freq='5T')), 
             np.append(electric_consumption_data[electric_consumption_data['Date'] >= electric_consumption_data['Date'].max() - datetime.timedelta(days=1)]['Value'], regression_wavelet_forecast[:, 0]), 
             label='Wavelet Forecast', color='red', linestyle='--')

    plt.plot(np.append(electric_consumption_data[electric_consumption_data['Date'] >= electric_consumption_data['Date'].max() - datetime.timedelta(days=1)]['Date'], pd.date_range(start=electric_consumption_data['Date'].max(), periods=forecast_steps, freq='5T')), 
             np.append(electric_consumption_data[electric_consumption_data['Date'] >= electric_consumption_data['Date'].max() - datetime.timedelta(days=1)]['Value'], hilbert_huang_forecast), 
             label='Hilbert Huang Transformation Forecast', color='blue', linestyle='dotted')
    
    plt.plot(np.append(electric_consumption_data[electric_consumption_data['Date'] >= electric_consumption_data['Date'].max() - datetime.timedelta(days=1)]['Date'], pd.date_range(start=electric_consumption_data['Date'].max(), periods=forecast_steps, freq='5T')),
                np.append(electric_consumption_data[electric_consumption_data['Date'] >= electric_consumption_data['Date'].max() - datetime.timedelta(days=1)]['Value'], arima_forecast),
                label='ARIMA Forecast', color='black', linestyle='dashed')

    plt.plot(electric_consumption_data[electric_consumption_data['Date'] >= electric_consumption_data['Date'].max() - datetime.timedelta(days=1)]['Date'], 
             electric_consumption_data[electric_consumption_data['Date'] >= electric_consumption_data['Date'].max() - datetime.timedelta(days=1)]['Value'], 
             label='Past', color='green')

    plt.title('Electric Consumption Price Forecasting Comparison')
    plt.xlabel('Date')
    plt.ylabel('Energy Price')
    plt.legend()
    plt.show()

# Visualize forecasts with actual values and past context
def visualize_forecasts_with_actuals(electric_consumption_data, regression_wavelet_forecast, hilbert_huang_forecast, arima_forecast, actual_values, forecast_steps):
    plt.figure(figsize=(22, 14))
    forecast_dates = pd.date_range(start=electric_consumption_data['Date'].max(), periods=forecast_steps, freq='5T')
    
    if len(forecast_dates) != len(actual_values):
        raise ValueError(f"forecast_dates and actual_values must have the same length, but have lengths {len(forecast_dates)} and {len(actual_values)} respectively.")
    
    past_dates = electric_consumption_data['Date'][-20:]  # Last 20 steps
    past_values = electric_consumption_data['Value'][-20:]  # Last 20 steps
    
    plt.plot(np.append(past_dates, forecast_dates[:4]), 
             np.append(past_values, regression_wavelet_forecast[:4, 0]), 
             label='Wavelet Forecast', color='red', linestyle='--')

    plt.plot(np.append(past_dates, forecast_dates[:4]), 
             np.append(past_values, hilbert_huang_forecast[:4]), 
             label='Hilbert Huang Transformation Forecast', color='blue', linestyle='dotted')
    
    plt.plot(np.append(past_dates, forecast_dates[:4]),
             np.append(past_values, arima_forecast[:4]),
             label='ARIMA Forecast', color='black', linestyle='dashed')

    plt.plot(past_dates, past_values, label='Past', color='green')
    plt.plot(forecast_dates[:4], actual_values[:4], label='Actual Values', color='orange', linestyle='solid')

    plt.title('Forecasts vs Actual Values with Past Context')
    plt.xlabel('Date')
    plt.ylabel('Energy Price')
    plt.legend()
    plt.show()