import pandas as pd
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import numpy as np
from dateutil.relativedelta import relativedelta
model = Prophet()
btc_data = yf.download("BTC-USD", start='2020-01-01', end='2024-06-10')
btc_data.reset_index(inplace=True)
btc_data = btc_data[['Date', 'Close']]
btc_data.columns = ['ds', 'y']
model = Prophet()
model.fit(btc_data)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
model.plot(forecast)
print((btc_data))
plt.show()