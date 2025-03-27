import numpy as np
from .func import load_obj

def get_stock_info():
    import pandas as pd
    df = pd.read_csv('../data/stock_basic.csv')
    stock_info = dict()
    for index, data in df.iterrows():
        stock_info[data['ts_code']] = {key: data[key] for key in data.keys() if key != 'ts_code'}
    return stock_info

def reprocess_data(data_path, max_price=10000):
    data_info = load_obj(data_path)
    update_data_info = dict()
    for key, value in data_info.items():
        price = value['value']
        if price.shape[-1] < 60:
            continue
        new_price = (price - price.min()) * max_price / (price.max() - price.min())
        update_data_info[key] = dict(
            value=new_price.astype(np.int32),
            vol_rate = value['vol_rate']
        )
    return update_data_info
