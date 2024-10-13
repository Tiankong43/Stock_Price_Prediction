import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as op
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error

plt.rcParams['font.family'] = ['SimHei']
save = './result/new2500.png'
class StockData(Dataset):
    def __init__(self, csv_file='./data/train/AAPL.csv', window_size=60, target_days_ahead=10):
        data = pd.read_csv(csv_file).values[:, 1:]
        self.len = len(data) - window_size - target_days_ahead + 1

        self.y_max = data[:, 3].max()
        self.y_min = data[:, 3].min()

        data = self.normalise(data) + 1e-5

        self.X = []
        self.y = []
        for i in range(window_size, self.len + window_size):
            self.X.append(data[i - window_size:i, :].astype(np.float32))
            self.y.append(data[i + target_days_ahead - 1, 3])

        self.y = np.array(self.y).astype(np.float32)

        self.X = torch.tensor(self.X)
        self.y = torch.tensor(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

    def denormalise(self, normalised_data):
        data = normalised_data.copy().reshape(-1, 1)
        original_range = self.y_max - self.y_min
        data = data * original_range + self.y_min
        return data.reshape(-1)

    def normalise(self, data):
        data = data.T
        for i in range(len(data)):
            data_min = data[i].min()
            data_max = data[i].max()
            data[i] = (data[i] - data_min) / (data_max - data_min)
        return data.T


class Net(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, output_size=1, layers=2, dropout_rate=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size * 2, output_size)
        self.function = torch.sigmoid

    def forward(self, X):
        X, _ = self.lstm(X, None)
        X = X[:, -1, :]
        X = self.dropout(X)
        X = self.linear(X)
        X = self.function(X)
        return X


stock_data = StockData()
data = DataLoader(dataset=stock_data, batch_size=5, shuffle=False)

EPCHO = 200

net = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
# model = torch.load('model5.pkl', map_location=device)

criteria = nn.MSELoss()
optimiser = op.Adam(net.parameters(), lr=0.01)
# scheduler = lr_scheduler.StepLR(optimiser, step_size=40, gamma=0.5)
scheduler = CosineAnnealingLR(optimiser, T_max=EPCHO, eta_min=1e-6)

losses = []

for epcho in range(EPCHO):
    epoch_losses = []
    epoch_mse = []
    epoch_mae = []
    epoch_msle = []
    for i, (X, y) in enumerate(data):
        X, y = X.to(device), y.to(device)
        optimiser.zero_grad()
        predict = net(X)

        mse = criteria(predict, y.unsqueeze(-1))
        mae = mean_absolute_error(predict.detach().cpu().numpy(), y.detach().cpu().numpy())
        msle = mean_squared_log_error(predict.detach().cpu().numpy(), y.detach().cpu().numpy())

        loss = criteria(predict, y.unsqueeze(-1))
        loss.backward()
        optimiser.step()
        epoch_losses.append(loss.item())
        epoch_mse.append(mse.item())
        epoch_mae.append(mae)
        epoch_msle.append(msle)

    scheduler.step()
    epoch_loss = np.mean(epoch_losses)
    losses.append(epoch_loss)
    epoch_avg_mse = np.mean(epoch_mse)
    epoch_avg_mae = np.mean(epoch_mae)
    epoch_avg_msle = np.mean(epoch_msle)

    # print('Epoch: {}.......... loss is {}'.format(epcho, epoch_loss))
    print(f'Epoch: {epcho+1}.......... loss: mse: {epoch_avg_mse:.5f}, mae: {epoch_avg_mae:.5f}, msle: {epoch_avg_msle:.5f}')


fig_width_px = 1200
fig_height_px = 650
fig_width_inch = fig_width_px / 100
fig_height_inch = fig_height_px / 100
plt.figure(figsize=(fig_width_inch, fig_height_inch))
plt.plot(losses, label='训练损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./result/newloss2500.png')
plt.show()


predict = np.array([])
actual = np.array([])


with torch.no_grad():
    for X, y in data:
        X, y = X.to(device), y.to(device)
        predict = np.append(predict, net(X).cpu().detach().numpy())
        actual = np.append(actual, y.cpu().detach().numpy())


predict = stock_data.denormalise(predict)
actual = stock_data.denormalise(actual)

print('------训练完成-------')
# torch.save(net, 'model5.pkl')

fig_width_px = 1200
fig_height_px = 650
fig_width_inch = fig_width_px / 100
fig_height_inch = fig_height_px / 100

plt.figure(figsize=(fig_width_inch, fig_height_inch))

plt.plot(predict, label='预测数据')
plt.plot(actual, label='实际数据')
plt.xlabel('日期')
plt.ylabel('收盘价')
plt.title('模型预测效果')
plt.legend()
plt.savefig(save)
plt.show()