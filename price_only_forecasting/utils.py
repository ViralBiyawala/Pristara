import datetime

# Print forecasts and actual values
def print_forecasts_and_actuals(cdate, regression_wavelet_forecast, hilbert_huang_forecast, arima_forecast, ans, forecast_steps):
    
    temp = cdate
    print("-> Wavelet Transform Forecast:")
    for i in range(forecast_steps):
        temp = temp + datetime.timedelta(minutes=5)
        print(f'\tDate: {temp} : {regression_wavelet_forecast[i,0]}  , Actual Price Observed: {ans[i]}')

    print("\n")

    temp = cdate
    print("-> Hilbert Huang Transformation Forecast:")
    for i in range(forecast_steps):
        temp = temp + datetime.timedelta(minutes=5)
        print(f'\tDate: {temp} : {hilbert_huang_forecast[i]} ,  Actual Price Observed: {ans[i]}')

    print("\n")

    temp = cdate
    print("-> ARIMA Forecast:")
    for i, (index, value) in enumerate(arima_forecast.items()):
        temp = temp + datetime.timedelta(minutes=5)
        try:
            print(f'\tDate: {temp} : {value} ,  Actual Price Observed: {ans[i]}')
        except KeyError as e:
            print(f'\tDate: {temp} : Forecast data not available, Actual Price Observed: {ans[i]} (KeyError: {e})')

    print("\n")

# Calculate and print statistics
def calculate_and_print_statistics(data):
    stder = data['Value'].sem()
    meanda = data['Value'].mean()
    print("Standard Error:", stder)
    print("Mean Value:", meanda)
    print("Upper Bound (Mean + 1.65 * Standard Error):", meanda + (1.65 * stder))
    print("Lower Bound (Mean - 1.65 * Standard Error):", meanda - (1.65 * stder))