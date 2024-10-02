
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

btc = yf.download('BTC-USD', start='2016-01-01', end='2024-10-01')
data = btc[['Close']]
data = data.dropna()
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)
def create_sequences(data):
    sequences = []
    targets = []
    for i in range(len(data) - 60):
        seq = data[i:i+60]
        target = data[i+60]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

x, y = create_sequences(scaled_data)
print(x)
print('asdasdasd')
print(y)
dates = btc.index[60:]
x_train, x_test = x[:2508], x[2508:]
y_train, y_test = y[:2508], y[2508:]
class set_data:
    def __init__(self, sequence, target):
        self.sequence = torch.tensor(sequence, dtype=torch.float32)
        self.targets = torch.tensor(target, dtype=torch.float32)
    def __len__(self):
        return len(self.sequence)
    def __getitem__(self, idx):
        return self.sequence[idx], self.targets[idx]
trn_dataset = set_data(x_train, y_train)
tst_dataset = set_data(x_test, y_test)
trnldr = DataLoader(trn_dataset, batch_size=64, shuffle=True)
tstldr = DataLoader(tst_dataset, batch_size=64, shuffle=False)
class _LSTM(nn.Module):
    def __init__(self):
        super(_LSTM, self).__init__()
        self.hid_sze = 50
        self.num_layer = 2
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        h_ = torch.zeros(self.num_layer, x.size(0), self.hid_sze).to(x.device)
        c_ = torch.zeros(self.num_layer, x.size(0), self.hid_sze).to(x.device)
        output, (h_, c_) = self.lstm(x, (h_, c_))
        output = output[:, -1, :]
        output = self.fc(output)
        
        return output
model = _LSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
for asd in range(50):
    model.train()
    for seq, tar in trnldr:
        seq = seq.to(device)
        tar = tar.to(device)
        output = model(seq)
        loss = criterion(output, tar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
model.eval()
pre, act = [], []
with torch.no_grad():
    for sequence, targets in tstldr:
        sequence = sequence.to(device)
        targets = targets.to(device)
        outputs = model(sequence)
        pre.append(outputs.cpu().numpy())
        act.append(targets.cpu().numpy())
pre = np.concatenate(pre).flatten()
act = np.concatenate(act).flatten()
pre = scaler.inverse_transform(pre.reshape(-1, 1)).flatten()
act = scaler.inverse_transform(act.reshape(-1, 1)).flatten()

rmse = np.sqrt(mean_squared_error(act, pre))
mae = mean_absolute_error(act, pre)
plt.figure(figsize=(14,7))
plt.plot(act, label='Actual')
plt.plot(pre, label='Predicted')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()