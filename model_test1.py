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
    asd = btc.idxmax()
    qwe = filtered_forecast.idxmax()
    print(asd, qwe)
evaluate_accuracy('2023-09-10')