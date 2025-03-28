import pickle
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']


def save_obj(obj, file_path):
    buf = pickle.dumps(obj)
    with open(file_path, "wb") as f:
        f.write(buf)


def load_obj(file_path):
    with open(file_path, "rb") as f:
        obj = pickle.loads(f.read())
    return obj


def draw_candlestick(start_date, values,  save_path="pred.png", title=None):

    data = {
        'Date': pd.date_range(start=start_date, periods=len(values), freq='B'),
        'Open': values[:, 0],
        'Close': values[:, 1],
        'Low': values[:, 2],
        'High': values[:, 3],
    }

    custom_style = mpf.make_mpf_style(
        base_mpf_style='yahoo',
        marketcolors=mpf.make_marketcolors(
            up='r',  # 上涨颜色为绿色
            down='g',  # 下跌颜色为红色
            edge='inherit',  # 边框颜色继承
        )
    )

    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    mpf.plot(df, type='candle', volume=False, style=custom_style, savefig=save_path, title=title, figscale=3)
