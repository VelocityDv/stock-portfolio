import numpy as np
import math

import matplotlib as plt 

from scipy.optimize import fsolve

class Portfolio:
    def __init__(self, df):
        self.df = df
        self.total_mean_return = []
        self.stock_returns = []
        
        self.mean = np.empty(0)
        self.covariance = np.empty(0)

        self.asset_tickers = []        
        # number of stocks n portfolio
        self.N = None

        self.consts = []

    def find_stocks(self):
        x = self.df.columns[1:]
        self.asset_tickers = x.tolist()
        self.N = len(self.asset_tickers)
        
    def returns(self):
        for column in self.df.columns:
            if column != "Date":
                stock_return = []
                i = 0
                s_prev = 0
                for s_cur in self.df[column]:
                    if i != 0:
                        stock_return.append((s_cur - s_prev) / s_prev)
                        s_prev = s_cur
                    else:
                        s_prev = s_cur
                    i += 1

                self.total_mean_return.append(np.mean(stock_return))
                self.stock_returns.append(stock_return)

        self.mean = np.round(np.array(self.total_mean_return).reshape(-1, 1), 5)
        self.covariance = np.round(np.cov(self.stock_returns), 5)
#         self.mean = np.array([[1], [2], [0]])
#         self.covariance = np.array([[1,0,0],[0,2,0],[0,0,1]])

    def specific_optimal_portfolio(self, t, money, can_short_sell, graph):
        c_inv = np.linalg.inv(self.covariance)
        e = np.ones((self.N, 1))
        
        a = np.dot(np.dot(e.T, c_inv), e)
        b = np.dot(np.dot(self.mean.T, c_inv), e)
        c = np.dot(np.dot(self.mean.T, c_inv), self.mean)
        d = (a * c) - (b * b)

        alpha = (1 / a) * np.dot(c_inv, e)
        beta = np.dot(c_inv, self.mean - ((b / a) * e))

        x = alpha + (t * (beta))

        self.consts = [a, b, c, d]
        
        if can_short_sell:
            allocation = money * x
            
            mean = ((b + (d * t)) / a) * money
            
            variance = (1 + (d * (t**2))) / a
            sd = math.sqrt(variance) * money
            
            for i in range(len(self.asset_tickers)):
                print(f"Allocation of money "
                    f"{self.asset_tickers[i]}: {allocation[i][0]:.5f}")
        
            print(f"mean return: {mean[0][0]:.5f}")      
            
            print(f"risk: {sd:.5f}")
            
        else:
            new_mean_returns = []
            new_covariance_returns = []
            dropped_index = []
            
            for i in range(len(x)):
                if x[i] < 0:
                    dropped_index.append(i)

            if len(dropped_index) == 0:
                new_mean_returns = self.mean
                new_covariance_returns = self.covariance
            else:           
                offset = 0
                for index in dropped_index:
                    if index != 0:
                        index -= offset
                    new_mean_returns = np.array(np.delete(self.mean, index))
                    
                    new_covariance_returns = np.delete(
                        self.covariance, index, axis = 0
                    )
                    new_covariance_returns = np.delete(
                        new_covariance_returns, index, axis = 1
                    )
                    offset += 1
                    
        
                new_mean_returns = new_mean_returns.reshape(-1, 1)              
                    
                c_inv_hat = np.linalg.inv(new_covariance_returns)
                e_hat = np.ones((len(new_mean_returns), 1))
            
                a_hat = np.dot(np.dot(e_hat.T, c_inv_hat), e_hat)
                b_hat = np.dot(np.dot(new_mean_returns.T, c_inv_hat), e_hat)
                c_hat = np.dot(np.dot(new_mean_returns.T, c_inv_hat), new_mean_returns)
                d_hat = (a_hat * c_hat) - (b_hat * b_hat)

                alpha_hat = (1 / a_hat) * np.dot(c_inv_hat, e_hat)
                beta_hat = np.dot(
                    c_inv_hat, new_mean_returns - ((b_hat / a_hat) * e_hat)
                )

                x_hat = alpha_hat + (t * beta_hat)

                allocation = money * x_hat
                mean = ((b_hat + (d_hat * t)) / a_hat) * money
            
                variance = (1 + (d_hat * t**2)) / a_hat
                sd = math.sqrt(variance) * money
                
                for index in dropped_index:
                    allocation = np.insert(allocation, index, [[0.0]], axis=0)
                
                for i in range(len(self.asset_tickers)):
                    print(f"Allocation of money "
                        f"{self.asset_tickers[i]}: {allocation[i][0]:.5f}")

                print(f"mean return: {mean[0][0]:.5f}")

                print(f"risk: {sd:.5f}")

        if graph: 
            # ---------- plotting ------------
            # The portfolio assets
            # The optimal (unconstrained) portfolio for the 
            # The minimum risk portfolio
            # The efficient frontier and minimum variance frontier
                    
            mu = []
            sig = []

            # these values are just a b c d 
            optimal_a = self.consts[0]
            optimal_b = self.consts[1]
            optmal_c = self.consts[2]
            optimal_d = self.consts[3]

            plt.figure(figsize=(10, 8))

            # portfolio assets
            for i in range(self.N):
                mu.append(self.mean[i][0])
                sig.append(math.sqrt(self.covariance[i][i]))

            plt.scatter(
                sig, mu, 
                color = "blue", 
                marker = "o", 
                label = "Portfolio asset (σ, μ)"
            )

            for i, asset_ticker in enumerate(self.asset_tickers):
                plt.annotate(asset_ticker, (sig[i], mu[i]))
    

            # optimal portfolio
            optimal_mean = (optimal_b[0][0] + (optimal_d[0][0] * t)) / optimal_a[0][0]
            optimal_variance = (1 + (optimal_d[0][0] * (t ** 2))) / optimal_a[0][0]
            
            plt.scatter(
                np.sqrt(optimal_variance), optimal_mean, 
                color="g", 
                marker="o", 
                label="Optimal Portfolio when t = 0.04"
            )
            plt.annotate("optimal portfolio", (np.sqrt(optimal_variance), optimal_mean))

        
            # MRP
            sig_mrp = 1 / np.sqrt(optimal_a[0][0])
            mu_mrp = optimal_b[0][0] / optimal_a[0][0]

            plt.scatter(
                sig_mrp, mu_mrp, 
                label = 'Minimum risk portfolio', 
                marker = "o", 
                color = "m"
            )
            plt.annotate("MRP", (sig_mrp, mu_mrp))
            

            # EF and MVF
            x = np.linspace(0, 0.02, 300)

            y_pos = (
                optimal_b[0][0]/optimal_a[0][0] + 
                np.sqrt((x ** 2 - (1/optimal_a[0][0])) * 
                (optimal_d[0][0]/optimal_a[0][0]))
            )
            
            y_neg = (
                optimal_b[0][0]/optimal_a[0][0] - 
                np.sqrt((x ** 2 - (1/optimal_a[0][0])) * 
                (optimal_d[0][0]/optimal_a[0][0]))
            )

            plt.plot(x, y_pos, label = 'Efficient frontier', color = 'black')
            
            plt.plot(
                x, y_neg, 
                label = 'Minimum variance frontier (red and black)', 
                color = 'r'
            )
            
            plt.xlabel('Risk (σ)')
            plt.ylabel('Return (μ)')
            plt.title('Portfolio')
            plt.legend()
                
                
    def optimal_portfolio_with_risk_free_asset(self, t, return_risk_free_asset, graph):
        
        x_hat = []
        
        c_inv = np.linalg.inv(self.covariance)
        e = np.ones((self.N, 1))
        
        r_bar = self.mean - (return_risk_free_asset * e)
        
        c_bar = np.dot(np.dot(r_bar.T, c_inv), r_bar)
        
        
        risky_asset = t * np.dot(c_inv, r_bar) 
        
        risk_free_asset = 1 - t * np.dot(e.T, np.dot(c_inv, r_bar))
        
        x_hat.append(risk_free_asset[0][0])
        
        for asset in risky_asset:
            x_hat.append(asset[0])
        
        print("New optimal portfolio")
        for i in range(len(self.asset_tickers) + 1):
            if i != 0:
                print(f"{self.asset_tickers[i-1]}: {x_hat[i]:.5f}")
            else:
                print(f"Risk free asset: {x_hat[i]:.5f}")


        # market portfolio 
        initial_guess = 0.01
        sig_market_portfolio = fsolve(market_portfolio, initial_guess)
        
        mu_market_portfolio = (
            return_risk_free_asset + 
        (np.sqrt(c_bar[0][0]) * 
            sig_market_portfolio)
        )

        # ------------ calculating betas ------------
        # using CAPM model to estimate 
        # Rk = ak + βkRM
        # βk = (r - r0) / (rm - r0)
        beta = []
        
        for mean_return in self.mean:
            beta.append(
                (mean_return[0] - return_risk_free_asset) / 
                (mu_market_portfolio[0] - return_risk_free_asset)
            )
        
        print("Betas")
        for i in range(len(self.asset_tickers)):
            print(f"{self.asset_tickers[i]}: {beta[i]:.5f}")
        

        if graph:
            # -------------- plotting -----------------
            # The assets in portfolio
            # The optimal unconstrained portfolios for the investor with only risky assets 
                # and with a risk-free asset 
            # The efficient frontier and minimum variance frontier
            # The capital market line
            # The market portfolio
                    
            mu = []
            sig = []

            optimal_a = self.consts[0]
            optimal_b = self.consts[1]
            optmal_c = self.consts[2]
            optimal_d = self.consts[3]

            plt.figure(figsize=(10, 8))

            # assets
            for i in range(self.N):
                mu.append(self.mean[i][0])
                sig.append(math.sqrt(self.covariance[i][i]))

            plt.scatter(sig, mu, color = "blue", marker = "o", label = "Portfolio assets")

            for i, asset_ticker in enumerate(self.asset_tickers):
                plt.annotate(asset_ticker, (sig[i], mu[i]))


            # portfolio with risky asset + risk free asset
            optimal_mean = (optimal_b[0][0] + (optimal_d[0][0] * t)) / optimal_a[0][0]
            optimal_variance = (1 + (optimal_d[0][0] * (t ** 2))) / optimal_a[0][0]
            
            plt.scatter(
                np.sqrt(optimal_variance), 
                optimal_mean, 
                color="g", 
                marker="o", 
                label="Optimal unconstrained Portfolio"
            )
            plt.annotate("risky asset only", (np.sqrt(optimal_variance), optimal_mean))
            
            mu_hat = return_risk_free_asset + t * c_bar
            
            variance_hat = c_bar * (t**2)
            sig_hat = np.sqrt(variance_hat)
            
            plt.scatter(sig_hat, mu_hat, color = 'c', label = "Optimal Portfolio with risk-free asset")
            plt.annotate("with risk free asset", (sig_hat, mu_hat))


            # EF and MVF
            x = np.linspace(0, 0.02, 300)
            
            y_pos = (
                optimal_b[0][0] / optimal_a[0][0] + 
                np.sqrt((x ** 2 - (1 / optimal_a[0][0])) * 
                (optimal_d[0][0] / optimal_a[0][0]))
            )
            
            y_neg = (
                optimal_b[0][0] / optimal_a[0][0] - 
                np.sqrt((x ** 2 - (1 / optimal_a[0][0])) * 
                (optimal_d[0][0] / optimal_a[0][0]))
            )
            plt.plot(x, y_pos, color = 'black')
            plt.plot(x, y_neg, label = 'Minimum variance frontier', color = 'black')

            
            # CML
            cml = return_risk_free_asset + (np.sqrt(c_bar[0][0]) * x)
            
            plt.plot(x, cml, label = 'Capital Market Line', color = 'red')
            
            market_portfolio = lambda sig_hat: (return_risk_free_asset + \
                                                (np.sqrt(c_bar[0][0]) * sig_hat)) - \
                                                (optimal_b[0][0] / optimal_a[0][0] + \
                                                np.sqrt((sig_hat ** 2 - \
                                                (1 / optimal_a[0][0])) * \
                                                (optimal_d[0][0] / optimal_a[0][0])))
    
            # market portfolio
            plt.scatter(
                sig_market_portfolio, mu_market_portfolio, 
                label = "Market portfolio", 
                color = 'm'
            )
            plt.annotate("market portfolio", (sig_market_portfolio, mu_market_portfolio))
            
            plt.xlabel('Risk (σ)')
            plt.ylabel('Return (μ)')
            plt.title('Portfolio')
            plt.legend()      
      
        
      
                

            

        