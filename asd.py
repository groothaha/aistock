import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
btc = yf.download('BTC-USD', start='2016-01-01', end='2024-10-01')
data = btc[['Close']]
data = data.dropna()
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)
def create_sequences(data, sequence_length=60):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i+sequence_length]
        target = data[i+sequence_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)
x, y = create_sequences(scaled_data)
dates = btc.index[60:]
x_train, y_train = x[:-1], y[:-1]
x_train = x_train.reshape(-1, 60, 1)
last_sequence = scaled_data[-60:].reshape(1, 60, 1)

class set_data(Dataset):
    def __init__(self, sequence, target):
        self.sequence = torch.tensor(sequence, dtype=torch.float32)
        self.targets = torch.tensor(target, dtype=torch.float32)
    
    def __len__(self):
        return len(self.sequence)
    
    def __getitem__(self, idx):
        return self.sequence[idx], self.targets[idx]
trn_dataset = set_data(x_train, y_train)
trnldr = DataLoader(trn_dataset, batch_size=64, shuffle=True)
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
for epoch in range(50):
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
future_predictions = []
current_sequence = torch.tensor(last_sequence, dtype=torch.float32).to(device)
with torch.no_grad():
    for _ in range(30):
        prediction = model(current_sequence)
        future_predictions.append(prediction.item())
        new_sequence = torch.cat((current_sequence[:, 1:, :], prediction.unsqueeze(2)), dim=1)
        current_sequence = new_sequence
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'])
future_dates = pd.date_range(start='2024-10-02', periods=30)
plt.plot(future_dates, future_predictions)
plt.xlim(pd.Timestamp('2020-01-01'), pd.Timestamp('2024-10-31'))
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()
