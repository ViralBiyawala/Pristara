import numpy as np
import pandas as pd
import datetime

# Load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, skiprows=4)
    df['DELIVERY_DATE'] = pd.to_datetime(df['DELIVERY_DATE'], format='%d-%m-%Y')
    df['DELIVERY_DATETIME'] = df['DELIVERY_DATE'] + pd.to_timedelta(df['DELIVERY_HOUR'] - 1, unit='h')
    df['DELIVERY_DATETIME'] += pd.to_timedelta((df['INTERVAL']) * 5, unit='m')
    return df

# Filter data
def filter_data(df, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df = df[df['DELIVERY_DATETIME'] > start_date]
    ans = df[(df['DELIVERY_DATETIME'] > end_date) & (df['DELIVERY_DATETIME'] <= end_date + datetime.timedelta(minutes=30))]
    ans = np.array(ans['ENGY_ON'])
    df = df[df['DELIVERY_DATETIME'] <= end_date]
    df.reset_index(drop=True, inplace=True)
    return df, ans

# Prepare data for visualization
def prepare_visualization_data(df):
    return pd.DataFrame({'Date': df['DELIVERY_DATETIME'], 'Value': df['ENGY_ON']})

# Load new data for comparison
def load_new_data(file_path, acdate):
    df = pd.read_csv(file_path, skiprows=4)
    df['DELIVERY_DATE'] = pd.to_datetime(df['DELIVERY_DATE'], format='%d-%m-%Y')
    df['DELIVERY_DATETIME'] = df['DELIVERY_DATE'] + pd.to_timedelta(df['DELIVERY_HOUR'] - 1, unit='h')
    df['DELIVERY_DATETIME'] += pd.to_timedelta((df['INTERVAL'] - 1) * 5, unit='m')
    df = df[df['DELIVERY_DATETIME'] > acdate]
    return df
