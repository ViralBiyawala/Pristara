import datetime
from data_processing import load_and_preprocess_data, filter_data, prepare_visualization_data, load_new_data
from visualization import visualize_data, visualize_forecasts, visualize_forecasts_with_actuals
from price_only_forecasting.wavelet import forecast_arima, forecast_hilbert_huang, forecast_wavelet_regression
from utils import print_forecasts_and_actuals, calculate_and_print_statistics

def main():
    # Load and preprocess data
    df = load_and_preprocess_data('../Data/On2_New.csv')
    
    forecast_steps = 4
    
    # Filter data
    df, ans = filter_data(df, '2023-12-20 00:00:00', '2023-12-22 00:00:00')
    
    # Prepare data for visualization
    electric_consumption_data = prepare_visualization_data(df)
    
    # Visualize the data
    # visualize_data(electric_consumption_data)
    
    
    # Perform ARIMA forecast
    arima_forecast = forecast_arima(electric_consumption_data, order=(7, 1, 1), forecast_steps=forecast_steps)
    
    # Perform Hilbert-Huang Transform forecast
    hilbert_huang_forecast = forecast_hilbert_huang(electric_consumption_data, forecast_steps=forecast_steps)
    
    # Perform wavelet regression forecast
    regression_wavelet_forecast = forecast_wavelet_regression(electric_consumption_data, wavelet='db4', level=0, forecast_steps=forecast_steps)
    # Visualize the forecasts
    # visualize_forecasts(electric_consumption_data, regression_wavelet_forecast, hilbert_huang_forecast, arima_forecast, forecast_steps)
    
    # Load new data for comparison
    cdate = electric_consumption_data['Date'].max()
    acdate = cdate
    df = load_new_data('../Data/On2_New.csv', acdate)
    cdate = cdate - datetime.timedelta(minutes=5)
    
    # Print forecasts and actual values
    print_forecasts_and_actuals(cdate, regression_wavelet_forecast, hilbert_huang_forecast, arima_forecast, ans[:forecast_steps], forecast_steps)

    # visualize forecasts with actual values
    # visualize_forecasts_with_actuals(electric_consumption_data, regression_wavelet_forecast, hilbert_huang_forecast, arima_forecast, ans[:forecast_steps], forecast_steps)
    
    # Calculate and print statistics
    calculate_and_print_statistics(electric_consumption_data)

if __name__ == "__main__":
    main()
