import pandas as pd

def merge_data(df_energy, df_gp, df_demand):
    df_merged = pd.merge(df_energy, df_demand, on='DELIVERY_DATETIME', how='left')
    df_merged = pd.merge(df_merged, df_gp, on='DELIVERY_DATETIME', how='outer')
    df_merged['ENGY_ON_D'].interpolate(method='linear', inplace=True)
    df_merged['Gas_Total'].interpolate(method='linear', inplace=True)
    df_merged['Hydro'].interpolate(method='linear', inplace=True)
    df_merged['Wind'].interpolate(method='linear', inplace=True)
    df_merged['Nuclear'].interpolate(method='linear', inplace=True)
    df_merged['close'].fillna(method='ffill', inplace=True)
    return df_merged

def filter_data(df, start_date, end_date):
    return df[(df['DELIVERY_DATETIME'] > start_date) & (df['DELIVERY_DATETIME'] <= end_date)]