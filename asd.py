import yfinance as yf
my_account = {}
balance = 10000000
# 한국 주식은 종목번호+.KS를 붙여야함
# example: 삼성전자는 '005930.KS'
class bs:
    def buy(stock, date, count):
        stock_data = yf.download(stock, start='2019-01-01', end='2023-12-31')
        price = stock_data.loc[date]
        asd = [stock, date, price, count]
        my_account[len(my_account)] = asd
    def sell(stock, date, count):
        for i in range(len(my_account)+1):
            if (my_account[i][0] == stock):
                stock_data = yf.download(stock, start='2019-01-01', end='2023-12-31')
                price_after = stock_data.loc[date]
                price_before = my_account[i][2]
                if (count > my_account[i][3]):
                    print('too many count')
                    break
                print(f"at {date} profits from the sale of stocks: {int(count*(price_after - price_before))}")
                break
            if (i == len(my_account)):
                print("there is no such stock")
                break
bs.buy('005930.KS', '2022-01-10', 1000)
bs.sell('005930.KS', '2023-01-10', 1000)
        

