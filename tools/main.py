import numpy as np
import tushare as ts
from stock_info import get_stock_info
from tqdm import tqdm
from func import save_obj
import time

token = "4f5f7d8d86bf6f75a5f27c14400ffd23d03d964859fa61dcc32a39dd"

pro = ts.pro_api(token)


def find_industry_stock(industry):
    stocks = dict()
    for stock_code in stock_info:
        data = stock_info[stock_code]
        if data['industry'] == industry:
            stocks[stock_code] = data
    return stocks


if __name__ == '__main__':

    stock_info = get_stock_info()

    params = ['open', 'close', 'high', 'low']
    data_info = dict()
    for stock_code in tqdm(stock_info):
        try:
            df = pro.daily(ts_code=stock_code)
        except Exception as error:
            print(str(error))
            time.sleep(5)
            continue
        value = np.array([df[k] for k in params])
        vol = np.array(df['vol'])
        stored_value = value[:, ::-1]
        stored_vol = vol[::-1]

        change_rate = stored_vol / stored_vol.mean()
        stock = stock_info[stock_code]
        data_info[stock['name']] = dict(value=stored_value, vol_rate=change_rate)

    save_obj(data_info, "data/data_info.db")
