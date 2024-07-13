import yfinance as yf
from datetime import date

class Data():
    def load_data(self):
        ticker_symbols = ['9988.HK', 'BABA']
        file_path = "data/alibaba.csv"

        start_date = '2019-11-26'
        end_date = date.today().strftime("%Y-%m-%d")

        data1 = yf.download(ticker_symbols[0], start=start_date, end=end_date)['Adj Close']
        data2 = yf.download(ticker_symbols[1], start=start_date, end=end_date)['Adj Close']
        # data.insert(0, 'Date', data.index)

        # data.to_csv(file_path, index=False)
        return data1, data2

# data = Data()
# data.load_data()
