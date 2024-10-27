import numpy as np
from sklearn.linear_model import LinearRegression
import pywt

# Wavelet decomposition and reconstruction functions
def wavelet_decomposition(data, wavelet='db4', level=4):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return coeffs

def wavelet_reconstruction(coeffs, wavelet='db4'):
    reconstructed_data = pywt.waverec(coeffs, wavelet)
    return reconstructed_data

# Wavelet regression forecasting
def forecast_wavelet_regression(data, wavelet='db4', level=1, forecast_steps=10):
    coeffs = wavelet_decomposition(data[['Value']], wavelet, level)
    model = LinearRegression()
    X = np.arange(1, len(coeffs[0]) + 1).reshape(-1, 1)
    extended_coeffs = []

    for coeff in coeffs:
        model.fit(X, coeff)
        future_X = np.arange(len(coeff) + 1, len(coeff) + 1 + forecast_steps).reshape(-1, 1)
        extrapolated_values = model.predict(future_X)
        extended_coeff = np.concatenate([coeff, extrapolated_values])
        extended_coeffs.append(extended_coeff)

    reconstructed_data = wavelet_reconstruction(extended_coeffs, wavelet)
    forecasted_values = reconstructed_data[-forecast_steps:]
    return forecasted_values