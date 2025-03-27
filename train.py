import torch
from model.dataset import StockDataset
from model.gpt import GPT, GPTConfig
from common import configure_optimizers, UpdateLr

if __name__ == "__main__":

    path = "data/process_data_info.db"

    device = "cuda"
    model = GPT(GPTConfig())
    model.to(device)
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))

    max_iters = 600000
    learning_rate = 6e-4
    log_interval = 100
    optimizer = configure_optimizers(model, learning_rate=learning_rate, device_type=device)
    update_lr = UpdateLr(optimizer, warmup_iters=2000, lr_decay_iters=max_iters, min_lr=6e-5, learning_rate=learning_rate)
    op = StockDataset(path, block_size=model.config.block_size, batch_size=12)

    for iter_num in range(max_iters):
        update_lr.update_lr(iter_num)
        x_values, x_vol_rates, y_values, y_vol_rates = op.gen_data()
        x_values, x_vol_rates = x_values.to(device), x_vol_rates.to(device)
        y_values, y_vol_rates = y_values.to(device), y_vol_rates.to(device)

        optimizer.zero_grad()
        loss = model(x_values, x_vol_rates, y_values, y_vol_rates)
        loss.backward()
        optimizer.step()

        if iter_num % log_interval == 0:
            print('Train iter: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                iter_num, max_iters,
                100. * iter_num / max_iters, loss.item()))
            torch.save(model.state_dict(), "model.pt")
