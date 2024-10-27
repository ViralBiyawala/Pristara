from PyEMD import EMD
from scipy.signal import hilbert
from sklearn.linear_model import LinearRegression
import numpy as np

# Hilbert-Huang Transform forecasting
def forecast_hilbert_huang(data, forecast_steps=10):
    predicted_value = []
    rp = np.array(data['Value'])
    model = LinearRegression()

    for i in range(forecast_steps):
        emd = EMD()
        IMFs = emd(rp)
        instantaneous_phase = []
        instantaneous_amplitude = []

        for imf in IMFs:
            analytic_signal = hilbert(imf)
            instantaneous_phase.append(np.unwrap(np.angle(analytic_signal)))
            instantaneous_amplitude.append(np.abs(analytic_signal))

        features = np.column_stack((instantaneous_phase[0][:-1], instantaneous_amplitude[0][:-1]))
        target = rp[1:]
        
        model.fit(features, target)

        next_features = np.column_stack((instantaneous_phase[0][-1:], instantaneous_amplitude[0][-1:]))
        predicted_value.append(model.predict(next_features)[0])
        rp = np.append(rp, predicted_value[-1])

    return predicted_value
