import sys
import numpy as np
import torch
from common import get_stock_code, get_stock_daily, draw_candlestick
from model.gpt import GPT, GPTConfig


class StockData:

    def __init__(self, value: np.ndarray, vol: np.ndarray, date: np.ndarray, max_price=10000):
        self.org_value = value
        self.org_vol = vol
        self.date = date
        self.value = None
        self.vol_rate = None
        self.max_price = max_price

    def preprocess_data(self):
        vol_rate = self.org_vol / self.org_vol.mean()
        price = self.org_value
        price_min, price_max = self.org_value.min(), self.org_value.max()
        new_price = (price - price_min) * self.max_price / (price_max - price_min)
        self.value = np.expand_dims(new_price.transpose(1, 0).astype(np.int32), axis=0)
        self.vol_rate = vol_rate[None, :, None]

        x_values = torch.from_numpy(self.value)
        x_vol_rates = torch.from_numpy(self.vol_rate).float()
        return x_values, x_vol_rates

    def postprocess_price(self, values):
        price_min, price_max = self.org_value.min(), self.org_value.max()
        new_price = (price_max - price_min) * (values - values.min()) / (values.max() - values.min())
        return new_price + price_min


if __name__ == "__main__":
    stock_names = ['洋河股份', '东山精密', '长城汽车', '立讯精密', '鹏鼎控股', '太阳纸业', '双汇发展']

    print("构建网络模型")
    model = GPT(GPTConfig())
    model.load_state_dict(torch.load('data/model.pt', map_location='cpu'))

    for stock_name in stock_names:
        print(f"预测{stock_name}")
        stock_code_info = get_stock_code()
        stock_code = stock_code_info[stock_name]
        block_size = 256
        daily = get_stock_daily(stock_code)

        params = ['open', 'close', 'high', 'low']
        value = np.array([daily[k][:block_size][::-1] for k in params])
        vol = np.array(daily['vol'])[:block_size][::-1]
        date = np.array(daily['trade_date'])[:block_size][::-1]

        pred_days = 1
        stock_data = StockData(value, vol, date)
        x_values, x_vol_rates = stock_data.preprocess_data()
        pred_x_values, pred_x_vol_rates, max_prob_list = model.generate(x_values, x_vol_rates, max_new_tokens=pred_days, top_k=10)
        res_x_values = pred_x_values[:, -pred_days:]

        values = stock_data.postprocess_price(pred_x_values)
        values = values.squeeze().cpu().numpy().round(2)

        start_date = date[0]
        max_prob = max_prob_list[0]
        start_date = "-".join([start_date[:4], start_date[4:6], start_date[6:]])
        draw_candlestick(start_date, values, f"{stock_name}.png", title=f"{stock_code}_{max_prob:.2f}")

