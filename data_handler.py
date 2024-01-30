import yfinance as yf
from datetime import date

class Data():
    def load_data(self):
        ticker_symbols = ['BHP.AX','CSL.AX','NAB.AX','TCL.AX','TLS.AX']


        start_date = '2022-07-01'
        end_date = date.today().strftime("%Y-%m-%d")

        data = yf.download(ticker_symbols, start=start_date, end=end_date)['Adj Close']

        data.insert(0, 'Date', data.index)

        file_path = "data/stock_data.csv"

        data.to_csv(file_path, index=False)
