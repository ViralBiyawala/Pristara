# **Pristara**
Welcome to **Pristara**, your go-to suite for advanced energy market forecasting. Whether you're dealing with fluctuating energy prices or consumption patterns, Pristara is designed to help you navigate these challenges with ease. Our project combines high-level signal decomposition, stochastic modeling, and hybrid machine learning to deliver top-notch predictions and insights.

In this repository, you'll find tools and methodologies for both **Price-Only Forecasting** and **Multi-Factor Forecasting**, each in their own modules to provide precise and actionable results.

## Key Features

### Data Management and Preprocessing

- **Data Loading**: Efficiently loads complex datasets from various sources, optimized for high-throughput and memory efficiency.
- **Data Transformation**: Cleans and processes noisy, irregular datasets using normalization, interpolation, and feature engineering, crucial for non-stationary energy data.

### Advanced Forecasting Techniques

#### Price-Only Forecasting

Focuses on time series data for energy prices using:
- **ARIMA Modeling**: Captures serial dependencies and residual patterns for precise price forecasting.
- **Hilbert Transform**: Adapts to real-time fluctuations by deriving instantaneous amplitude and phase.
- **Wavelet Packet Decomposition**: Breaks down price signals for improved resolution in non-stationary environments.

#### Multi-Factor Forecasting

Combines multiple data inputs to create a comprehensive model using:
- **Fourier Series Expansion**: Identifies seasonal trends in high-dimensional data.
- **Linear Regression**: Models the relationship between dependent and independent variables for trend analysis.
- **ARIMA**: Uses historical data to predict future trends.
- **Wavelet Multi-Resolution Analysis (MRA)**: Analyzes data at different scales to detect transient features.

### Visual Analytics and Data Representation

The **visualization.py** module offers:
- **Multi-axis Charts**: Combines price, consumption, and other variables for comparative analysis.
- **Spectral Density Plots**: Analyzes cyclic price patterns in the frequency domain.
- **Decomposition Diagrams**: Visualizes signal decompositions to highlight data patterns.

### Utility Functions and Tools

- **Auxiliary Scripts**: The **utils.py** script provides rapid data transformation, validation, and optimized computation.
- **Pipeline Orchestration**: Ensures smooth operation across forecasting models, allowing for easy customization and integration.

## Getting Started

Clone the repository and install dependencies:

```shell
git clone https://github.com/viralbiyawala/Pristara.git
cd Pristara
pip install -r requirements.txt
```

## Running Forecasts

To run a forecast simulation, go to either the **price_only_forecasting** or **multi_factor_forecasting** directory and start the main forecasting script:

```shell
cd price_only_forecasting
python Power_forecasting.py
```

## Visualization

Use the visualization functions in your analysis or a Jupyter notebook to dynamically interpret forecasting accuracy and model behavior.

## Technical Highlights

Pristara integrates cutting-edge forecasting and decomposition methods:
- **Empirical Mode Decomposition (EMD)**: Extracts core oscillatory modes to reveal hidden patterns.
- **Hilbert Transform and Instantaneous Frequency Extraction**: Enables adaptive, real-time forecasts.
- **Wavelet Packet Analysis**: Decomposes signals at multiple resolutions for detailed insights.
- **Machine Learning Integration**: Combines deep learning with traditional time series techniques for robust forecasting.

## Why Pristara?

Pristara goes beyond traditional forecasting by merging predictive analytics with advanced statistical modeling. It helps users make data-driven decisions to optimize energy usage, anticipate price changes, and reduce costs. The Hilbert-Huang Transform is particularly effective, offering a more adaptive and accurate approach compared to traditional modeling techniques.

## Contributing

We welcome contributions from developers and researchers. Check out our [contribution guide](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License. See the LICENSE file for more information.
