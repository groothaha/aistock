import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta
btc_data = yf.download('BTC-USD', start='2016-01-01', end='2024-09-30')
btc_data = btc_data[['Close']].reset_index()
btc_data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
btc_data['t'] = (btc_data['ds'] - btc_data['ds'].min()).dt.days
def model_g(t, beta):
    exponent = -beta * (t - 30000)
    exponent = np.clip(exponent, -700, 700)
    return 90000 / (1 + np.exp(exponent))

def model_s(t, P, N, a, b):
    seasonal = np.zeros(len(t))
    for i in range(1, N+1):
        seasonal += a[i-1] * np.sin(2 * np.pi * i * t / P) + b[i-1] * np.cos(2 * np.pi * i * t / P)
    return seasonal
def model_h(t, holidays, alpha):
    return np.array([alpha if date in holidays else 0 for date in t])

def prophet_model(t, beta, a, b, holidays, alpha, P, N):
    trend = model_g(t, beta)
    season = model_s(t, P, N, a, b)
    holiday = model_h(t, holidays, alpha)
    return trend + season + holiday
def loss(params, t, y, holidays, P, N):
    beta = params[0]          
    a = params[1:1+N]        
    b = params[1+N:1+2*N]      
    alpha = params[-1]         
    y_pred = prophet_model(t, beta, a, b, holidays, alpha, P, N)
    return np.mean((y - y_pred) ** 2)
np.random.seed(42)
initial_params = np.random.randn(1 + 2*10 + 1)
holiday_dates = [
    '2016-07-09', '2017-01-28', '2017-12-15', '2018-07-06',
    '2019-05-11', '2020-05-11', '2021-05-11', '2022-05-11',
    '2023-05-11', '2024-05-11'
]
holiday_days = (pd.to_datetime(holiday_dates) - btc_data['ds'].min()).days.astype(int).tolist()
result = minimize(
    loss, 
    initial_params, 
    args=(btc_data['t'].values, btc_data['y'].values, holiday_days, 365.25, 10), 
    method='L-BFGS-B'
)
optimized_params = result.x
beta_ = optimized_params[0]
a_ = optimized_params[1:1+10]
b_ = optimized_params[1+10:1+2*10]
alpha_ = optimized_params[-1]
last_day = btc_data['t'].max()
t_future = np.arange(last_day + 1, last_day + 1 + 365)
ds_future = pd.date_range(start=btc_data['ds'].max() + timedelta(days=1), periods=365)
t_combined = np.concatenate([btc_data['t'].values, t_future])
ds_combined = pd.concat([btc_data['ds'], pd.Series(ds_future)], ignore_index=True)
future_holidays = holiday_days.copy()
for day in range(last_day + 1, last_day + 1 + 365):
    day_of_year = day % 365
    if day_of_year in [hd % 365 for hd in holiday_days]:
        future_holidays.append(day)
y_pred_combined = prophet_model(t_combined, beta_, a_, b_, future_holidays, alpha_, 365.25, 10)
y_pred_past = y_pred_combined[:len(btc_data)]
y_pred_future = y_pred_combined[len(btc_data):]

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(btc_data['ds'], btc_data['y'], label='Actual')
plt.plot(btc_data['ds'], y_pred_past, label='Fitted', linestyle='--')
plt.plot(ds_future, y_pred_future, label='Forecast', linestyle='--')
plt.axvline(x=btc_data['ds'].max(), color='red', linestyle=':', label='Forecast Start')
plt.xlabel('Date')
plt.ylabel('bitcoin price')
plt.title('predict')
plt.legend()
plt.grid(True)
plt.show()
