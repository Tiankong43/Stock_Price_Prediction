import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta


def get_data(csv):
    path = csv
    class StockData(Dataset):
        def __init__(self, csv_file=path):
            data = pd.read_csv(csv_file).values[:, 1:]
            data2 = pd.read_csv(csv_file)
            self.date = data2['Date'].tolist()
            self.close = data2['Close'].tolist()
            self.date = self.date[-20:]
            self.close = self.close[-20:]
            data = data[-70:, :]
            self.len = len(data)
            self.X = []
            self.y_max = data[:, 3].max()
            self.y_min = data[:, 3].min()
            data = self.normalise(data) + 1e-5

            for i in range(10):
                self.X.append(data[i:i+59, :].astype(np.float32))

            self.X = torch.tensor(self.X)

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

        def __getitem__(self, index):
            return self.X[index]

        def __len__(self):
            return len(self.X)

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

    stock_data = StockData(path)
    actual_close = stock_data.close
    actual_date = stock_data.date
    data = DataLoader(dataset=stock_data, batch_size=5, shuffle=False)


    model = Net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('model5.pkl', map_location=device)

    model.eval()

    predict = np.array([])

    with torch.no_grad():
        for X in data:
            # print(X)
            X = X.to(device)
            predict = np.append(predict, model(X).cpu().detach().numpy())

    predict = stock_data.denormalise(predict)


    #
    # latest_date = datetime.strptime(actual_date[-1], '%Y-%m-%d')
    # for i in range(10):
    #     new_date = latest_date + timedelta(days=i + 1)
    #     new_date_str = new_date.strftime('%Y-%m-%d')
    #     actual_date.append(new_date_str)
    #
    # close = actual_close + predict.tolist()
    # close = [round(i, 2) for i in close]
    # pre_data = []
    # for i in range(len(close)):
    #     list = []
    #     list.append(actual_date[i])
    #     list.append(close[i])
    #     pre_data.append(list)

    base_date = datetime.strptime('2024-04-20', '%Y-%m-%d')
    target_days = 10
    future_dates = []
    future_closes = []

    current_date = datetime.strptime(actual_date[-1], '%Y-%m-%d')
    current_date += timedelta(days=1)

    for _ in range(target_days):
        days_since_base = (current_date - base_date).days
        if days_since_base % 7 == 0:
            current_date += timedelta(days=2)
        else:
            future_dates.append(current_date.strftime('%Y-%m-%d'))
            future_closes.append(round(predict[len(future_closes)], 2))
            current_date += timedelta(days=1)


    while len(future_dates) < target_days:
        future_dates.append(current_date.strftime('%Y-%m-%d'))
        future_closes.append(round(predict[len(future_closes)], 2))
        current_date += timedelta(days=1)

    close = actual_close + future_closes
    close = [round(i, 2) for i in close]
    date = actual_date + future_dates
    pre_data = []


    for i in range(len(close)):
        list = []
        list.append(date[i])
        list.append(close[i])
        pre_data.append(list)

    # pre_data = [[date, close] for date, close in zip(future_dates, future_closes)]

    print(pre_data)

    return pre_data

