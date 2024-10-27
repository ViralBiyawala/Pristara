import pandas as pd

def load_data():
    df_energy = pd.read_csv('../Data/On2_New.csv', skiprows=4)
    df_energy['DELIVERY_DATE'] = pd.to_datetime(df_energy['DELIVERY_DATE'], format='%d-%m-%Y')
    df_energy['DELIVERY_DATETIME'] = df_energy['DELIVERY_DATE'] + pd.to_timedelta(df_energy['DELIVERY_HOUR']-1, unit='h')
    df_energy['DELIVERY_DATETIME'] += pd.to_timedelta((df_energy['INTERVAL']) * 5, unit='m')
    df_energy.reset_index(drop=True, inplace=True)

    df_gp = pd.read_csv('../Data/gp.csv')
    df_gp['DELIVERY_DATETIME'] = pd.to_datetime(df_gp['time'], format='%Y-%m-%dT%H:%M:%S-05:00')
    df_gp = df_gp.drop(['open', 'time', 'high', 'low', 'Volume', 'Volume MA'], axis=1)

    df_demand = pd.read_csv('../Data/Demand_New.csv', skiprows=3)
    df_demand['DELIVERY_DATE'] = pd.to_datetime(df_demand['Date'], format='%d-%m-%Y')
    df_demand['DELIVERY_DATETIME'] = df_demand['DELIVERY_DATE'] + pd.to_timedelta(df_demand['Hour'], unit='h')
    df_demand.reset_index(drop=True, inplace=True)

    return df_energy, df_gp, df_demand