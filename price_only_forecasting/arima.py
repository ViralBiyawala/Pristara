from statsmodels.tsa.arima.model import ARIMA

# ARIMA forecasting function
def forecast_arima(data, order=(7, 1, 1), forecast_steps=10):
    model = ARIMA(data['Value'], order=order)
    fit_model = model.fit()
    forecasted_values = fit_model.forecast(steps=forecast_steps)
    return forecasted_values