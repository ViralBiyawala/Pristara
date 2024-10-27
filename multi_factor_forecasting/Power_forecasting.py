import datetime
from data_loading import load_data
from data_processing import merge_data, filter_data
from visualization import visualize_data, visualize_results, visualize_forecasts_with_actuals
import numpy as np
from forecasting import (
    fourier_series_analysis,
    inverse_fourier_series,
    predict_future_points,
    emd_hilbert_predict,
    forecast_arima,
    final_prediction
)
from sklearn.linear_model import LinearRegression

def main():
    df_energy, df_gp, df_demand = load_data()
    df_merged = merge_data(df_energy, df_gp, df_demand)
    df_merged = filter_data(df_merged, '12-20-2023 00:00:00', '12-22-2023 00:20:00')
    df_demand = filter_data(df_demand, '12-20-2023 00:00:00', '12-22-2023 00:00:00')
    df_energy = filter_data(df_energy, '12-20-2023 00:00:00', '12-22-2023 00:00:00')

    # Extract the last 4 values (20 minutes) for forecast checking with actual values
    actual_values = df_merged['ENGY_ON'].values[-4:]
    
    # Remove the last 4 values from the dataframe
    df_merged = df_merged.iloc[:-4]

    # visualize_data(df_merged)
    # print(df_merged.tail())

    sampling_rate = 10
    original_combined_signal = df_demand['ENGY_ON_D'].values
    frequencies, fft_values = fourier_series_analysis(original_combined_signal, sampling_rate)
    inverse_combined_signal = inverse_fourier_series(frequencies, fft_values)
    forecast_steps = 4
    n_points_to_predict = 1 * forecast_steps
    demand_forecasted = predict_future_points(frequencies, fft_values, len(original_combined_signal), sampling_rate, n_points_to_predict, original_combined_signal)
    demand_forecasted = np.interp(np.arange(0, len(demand_forecasted), 1/12), np.arange(0, len(demand_forecasted)), demand_forecasted)

    predicted_wind = emd_hilbert_predict(df_demand['Wind'], forecast_steps)
    predicted_nuclear = emd_hilbert_predict(df_demand['Nuclear'], forecast_steps)
    predicted_hydro = emd_hilbert_predict(df_demand['Hydro'], forecast_steps)

    gas_model = LinearRegression()
    gas_demand = np.array(df_demand['ENGY_ON_D'])
    gas_hydro = np.array(df_demand['Hydro'])
    gas_wind = np.array(df_demand['Wind'])
    gas_total = np.array(df_demand['Gas_Total'])
    predicted_gas = []
    for i in range(forecast_steps):
        X = np.column_stack((gas_demand[:-1], gas_hydro[:-1], gas_wind[:-1]))
        y = gas_total[:-1]
        gas_model.fit(X, y)
        X_test = np.column_stack((gas_demand[-1], gas_hydro[-1], gas_wind[-1]))
        predicted_gas.append(gas_model.predict(X_test)[0])
        gas_total = np.append(gas_total, predicted_gas[-1])
        gas_demand = np.append(gas_demand, demand_forecasted[i])
        gas_hydro = np.append(gas_hydro, predicted_hydro[i])
        gas_wind = np.append(gas_wind, predicted_wind[i])
    predicted_gas = np.interp(np.arange(0, len(predicted_gas), 1/12), np.arange(0, len(predicted_gas)), predicted_gas)

    predicted_values = final_prediction(df_merged, df_demand, demand_forecasted, predicted_hydro, predicted_wind, predicted_gas, predicted_nuclear, forecast_steps)

    # Visualize the forecasts with actual values
    visualize_forecasts_with_actuals(df_merged, predicted_values, actual_values, forecast_steps)

if __name__ == "__main__":
    main()