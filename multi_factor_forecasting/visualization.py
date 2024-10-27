import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def visualize_data(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['DELIVERY_DATETIME'], df['ENGY_ON'])
    plt.xlabel('Date')
    plt.ylabel('Consumption')
    plt.legend()
    plt.show()

def visualize_results(df_merged, predicted_values, n_hours_to_predict):
    plt.figure(figsize=(22, 14))
    plt.plot(np.append(df_merged[df_merged['DELIVERY_DATETIME'] >= df_merged['DELIVERY_DATETIME'].max() - pd.Timedelta(days=1)]['DELIVERY_DATETIME'], pd.date_range(start=df_merged['DELIVERY_DATETIME'].max(), periods=n_hours_to_predict, freq='5T')), np.append(df_merged[df_merged['DELIVERY_DATETIME'] >= df_merged['DELIVERY_DATETIME'].max() - pd.Timedelta(days=1)]['ENGY_ON'], predicted_values), label='Hilbert Huang Transformation Forecast', color='k', linestyle='--')
    plt.plot(df_merged[df_merged['DELIVERY_DATETIME'] >= df_merged['DELIVERY_DATETIME'].max() - pd.Timedelta(days=1)]['DELIVERY_DATETIME'], df_merged[df_merged['DELIVERY_DATETIME'] >= df_merged['DELIVERY_DATETIME'].max() - pd.Timedelta(days=1)]['ENGY_ON'], label='Past', color='orange')
    plt.title('Electric Consumption Price Forecasting')
    plt.xlabel('DELIVERY_DATETIME')
    plt.ylabel('Energy Price')
    plt.legend()
    plt.show()

# Visualize forecasts with actual values and past context
def visualize_forecasts_with_actuals(df_merged, predicted_values, actual_values, n_hours_to_predict):
    plt.figure(figsize=(22, 14))
    forecast_dates = pd.date_range(start=df_merged['DELIVERY_DATETIME'].max(), periods=n_hours_to_predict, freq='5T')
    
    if len(forecast_dates) != len(actual_values):
        raise ValueError(f"forecast_dates and actual_values must have the same length, but have lengths {len(forecast_dates)} and {len(actual_values)} respectively.")
    
    past_dates = df_merged['DELIVERY_DATETIME'][-20:]  # Last 20 steps
    past_values = df_merged['ENGY_ON'][-20:]  # Last 20 steps
    
    plt.plot(np.append(past_dates, forecast_dates[:4]), 
             np.append(past_values, predicted_values[:4]), 
             label='Predicted Values', color='red', linestyle='--')

    plt.plot(past_dates, past_values, label='Past', color='green')
    plt.plot(forecast_dates[:4], actual_values[:4], label='Actual Values', color='orange', linestyle='solid')
    plt.title('Forecasts vs Actual Values with Past Context')
    plt.xlabel('DELIVERY_DATETIME')
    plt.ylabel('Energy Price')
    plt.legend()
    plt.show()
