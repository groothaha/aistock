import pandas as pd
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import numpy as np
from dateutil.relativedelta import relativedelta
def evaluate_accuracy(end):
    start = datetime.strptime(end, '%Y-%m-%d') - relativedelta(months=3)
    start = start.strftime('%Y-%m-%d')
    btc_data = yf.download("BTC-USD", start='2016-01-01', end=start)
    btc_data.reset_index(inplace=True)
    btc_data = btc_data[['Date', 'Close']]
    btc_data.columns = ['ds', 'y']
    model = Prophet()
    model.fit(btc_data)
    future = model.make_future_dataframe(periods=2000)
    forecast = model.predict(future)
    filtered_forecast = forecast[(forecast['ds'] >= start) & (forecast['ds'] <= end)]['yhat'][:-1]
    btc = yf.download("BTC-USD", start=start, end=end)
    btc.reset_index(inplace=True)
    btc = btc[['Close']]
    asd = []
    qwe = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days
    for i in range(qwe):
        asd.append(abs(filtered_forecast.iloc[i] - btc.iloc[i]['Close']) / btc.iloc[i]['Close'] * 100)
    asd = np.array(asd)
    return np.mean(asd)
print(evaluate_accuracy('2024-07-10'))
_x = []
_y = []
for i in range(365):
    c_d = datetime.strptime('2023-06-10', '%Y-%m-%d') + timedelta(days=i)
    _x.append(i)
    asd = evaluate_accuracy(c_d.strftime('%Y-%m-%d'))
    _y.append(asd)
    print(i, asd)
plt.plot(_x, _y)
plt.show()