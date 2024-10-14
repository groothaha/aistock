import requests
import time
import datetime
import csv
import os

def get_bitcoin_price_binance():
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {'symbol': 'BTCUSDT'}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return float(data['price'])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
def save_to_csv(time_str, price):
    file_exists = os.path.isfile('bitcoin_prices_binance.csv')
    with open('bitcoin_prices_binance.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Time', 'Bitcoin Price (USD)'])
        writer.writerow([time_str, price])
def main():
    while True:
        price = get_bitcoin_price_binance()
        if price:
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{current_time}] 비트코인 가격(USD): {price}")
            save_to_csv(current_time, price)
        else:
            print("가격 정보를 가져오지 못했습니다.")
        
        time.sleep(60)

if __name__ == "__main__":
    main()
