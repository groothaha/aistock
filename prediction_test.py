import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
df = pd.read_csv('bitcoin.csv')
df['ds'] = pd.to_datetime(df['ds'])
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=180, freq='T')
forecast = model.predict(future)
fig = model.plot(forecast)
plt.xlabel("Date")
plt.ylabel("Price")
ax = fig.gca()
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

#plt.xticks(rotation=45)
plt.grid()
plt.show()
