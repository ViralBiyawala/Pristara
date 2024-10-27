import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from PyEMD import EMD
from scipy.signal import hilbert
from sklearn.linear_model import LinearRegression, Ridge
from scipy.fft import fft, ifft, fftfreq
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

def fourier_series_analysis(signal, sampling_rate):
    N = len(signal)
    frequencies = fftfreq(N, d=1/sampling_rate)
    fft_values = fft(signal)
    return frequencies, fft_values

def inverse_fourier_series(frequencies, fft_values):
    inverse_signal = ifft(fft_values)
    return np.real(inverse_signal)

def predict_future_points(frequencies, fft_values, num_points, sampling_rate, n, original_combined_signal):
    features = np.zeros((num_points, len(frequencies) * 2))
    for i in range(15):
        sine_function = np.sin(2 * np.pi * frequencies[i] * np.arange(num_points) / sampling_rate)
        cosine_function = np.cos(2 * np.pi * frequencies[i] * np.arange(num_points) / sampling_rate)
        features[:, i * 2] = fft_values[i].real * sine_function
        features[:, i * 2 + 1] = fft_values[i].real * cosine_function

    target = original_combined_signal[:num_points]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=38)
    model = LinearRegression()
    model.fit(X_train, y_train)
    future_features = features[-n:, :]
    predictions = model.predict(future_features)
    return predictions

def emd_hilbert_predict(data, forecast_steps):
    predicted_values = []
    residuals = np.array(data)
    model = LinearRegression()
    for i in range(forecast_steps):
        emd = EMD()
        imfs = emd(residuals)
        instantaneous_phase = []
        instantaneous_amplitude = []
        for imf in imfs:
            analytic_signal = hilbert(imf)
            instantaneous_phase.append(np.unwrap(np.angle(analytic_signal)))
            instantaneous_amplitude.append(np.abs(analytic_signal))
        features = np.column_stack((instantaneous_phase[0][:-1], instantaneous_amplitude[0][:-1]))
        target = residuals[1:]
        model.fit(features, target)
        next_features = np.column_stack((instantaneous_phase[0][-1:], instantaneous_amplitude[0][-1:]))
        predicted_values.append(model.predict(next_features)[0])
        residuals = np.append(residuals, predicted_values[-1])
    return np.interp(np.arange(0, len(predicted_values), 1/12), np.arange(0, len(predicted_values)), predicted_values)

def forecast_arima(data, order=(7, 1, 1), forecast_steps=10):
    model = ARIMA(data['ENGY_ON'], order=order)
    fit_model = model.fit()
    forecasted_values = fit_model.forecast(steps=forecast_steps)
    return forecasted_values

from sklearn.preprocessing import StandardScaler

def final_prediction(df_merged, df_demand, demand_forecasted, predicted_hydro, predicted_wind, predicted_gas, predicted_nuclear, n_hours_to_predict):
    demand_forecasted = demand_forecasted.reshape(-1, 1)
    predicted_values = []
    residuals = np.array(df_merged['ENGY_ON'])
    linear_model = Ridge(alpha=0.00001)
    
    for i in range(n_hours_to_predict):        
        min_length = min(len(df_demand['ENGY_ON_D'][:-1]), len(df_demand['Hydro'][:-1]), len(df_demand['Wind'][:-1]), len(df_demand['Gas_Total'][:-1]), len(df_demand['Nuclear'][:-1]), len(df_merged['close'][:-1]))
        
        features = np.column_stack((df_demand['ENGY_ON_D'][:min_length], df_demand['Hydro'][:min_length], df_demand['Wind'][:min_length], df_demand['Gas_Total'][:min_length], df_demand['Nuclear'][:min_length], df_merged['close'][:min_length]))
        target = residuals[1:min_length+1]
        
        linear_model.fit(features, target)
        
        next_features = np.column_stack((demand_forecasted[i], predicted_hydro[i], predicted_wind[i], predicted_gas[i], predicted_nuclear[i], df_merged['close'].iloc[-1]))
        predicted_values.append(linear_model.predict(next_features)[0])
        residuals = np.append(residuals, predicted_values[-1])
    
    return predicted_values