import numpy as np
import torch
import random
from common import load_obj


class StockDataset:

    def __init__(self, data_path, block_size, batch_size, max_price=10000):
        self.block_size = block_size
        self.batch_size = batch_size
        self.max_price = max_price
        self.data_info = load_obj(data_path)
        self.stock_names = list(self.data_info.keys())

    def gen_data(self):

        names = random.sample(self.stock_names, k=self.batch_size)
        values = [self.data_info[name]['value'] for name in names]
        vol_rates = [self.data_info[name]['vol_rate'] for name in names]

        min_block = min(len(v) for v in vol_rates)
        block_size = min(self.block_size, min_block)

        ix = [random.randint(0, len(v) - block_size) for v in vol_rates]
        x_values = np.stack([value[:, i:i+block_size-1] for value, i in zip(values, ix)], axis=0).transpose([0, 2, 1])
        x_vol_rates = np.stack([v[i:i+block_size-1] for v, i in zip(vol_rates, ix)], axis=0)

        y_values = np.stack([value[:, i+1:i+block_size] for value, i in zip(values, ix)], axis=0).transpose([0, 2, 1])
        y_vol_rates = np.stack([v[i+1:i+block_size] for v, i in zip(vol_rates, ix)], axis=0)

        x_values = torch.from_numpy(x_values)
        x_vol_rates = torch.from_numpy(x_vol_rates).unsqueeze(-1).float()
        y_values = torch.from_numpy(y_values)
        y_vol_rates = torch.from_numpy(y_vol_rates).unsqueeze(-1).float()
        return x_values, x_vol_rates, y_values, y_vol_rates

