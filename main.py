import pandas as pd 
import numpy as np
import math

from portfolio import Portfolio
from data_handler import Data

def main(): 
    data = Data()
    data.load_data()

    df = pd.read_csv('data/stock_data.csv')

    portfolio = Portfolio(df)
    portfolio.find_stocks()

    portfolio.returns()
    print("Mean Returns")
    for i in range(len(portfolio.asset_tickers)):
        print(f"{portfolio.asset_tickers[i]}: t =  {portfolio.mean[i][0]}")

    print(f"Covariance Matrix - C:\n {portfolio.covariance}")


    t = 0.04
    money = 10000

    print("Short selling allowed")
    portfolio.specific_optimal_portfolio(t, money, True, True )

    print("\nNo short selling allowed")
    portfolio.specific_optimal_portfolio(t, money, False, True)

    portfolio.optimal_portfolio_with_risk_free_asset(0.04, 0.0001)
    

if __name__ == "__main__":
    main()