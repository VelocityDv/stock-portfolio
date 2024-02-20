# realise IBKR needs pro account to request hisotrical data

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import *

import threading
import time
import pandas as pd

class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = {'Time': [], 'BABA_USD': [], 'BABA_HKD': []}

    def historicalData(self, reqId, bar):
        timestamp = pd.to_datetime(bar.date)
        self.data['Time'].append(timestamp)
        if reqId == 1:
            self.data['BABA_USD'].append(bar.close)
        elif reqId == 2:
            self.data['BABA_HKD'].append(bar.close)

def run_loop():
    app.run()

app = IBapi()
app.connect('127.0.0.1', 7497, 123)

api_thread = threading.Thread(target=run_loop, daemon=True)
api_thread.start()

time.sleep(1)

alibaba_contract_US = Contract()
alibaba_contract_US.symbol = 'BABA'
alibaba_contract_US.secType = 'STK'
alibaba_contract_US.exchange = 'SMART'
alibaba_contract_US.currency = 'USD'

alibaba_contract_HK = Contract()
alibaba_contract_HK.symbol = '9988'
alibaba_contract_HK.secType = 'STK'
alibaba_contract_HK.exchange = 'SEHK'
alibaba_contract_HK.currency = 'HKD'

end_date_time = (pd.Timestamp.now() - pd.DateOffset(days=7)).strftime("%Y%m%d %H:%M:%S")

app.reqHistoricalData(1, alibaba_contract_US, end_date_time, "1 W", "1 min", "TRADES", 1, 1, False, [])
app.reqHistoricalData(2, alibaba_contract_HK, end_date_time, "1 W", "1 min", "TRADES", 1, 1, False, [])

time.sleep(60)  

app.disconnect()

df = pd.DataFrame(app.data)
df.set_index('Time', inplace=True)

print(df.head())  

df.to_csv('data/baba_historical_data.csv', index=False)