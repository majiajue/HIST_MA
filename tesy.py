import pandas as pd
import pickle
import tushare as ts
# 初始化pro接口
pro = ts.pro_api('7bee558347c0922dfd601254294a5726654493fdf5545d5c28ee2153')
df = pro.daily_basic(**{
    "ts_code": "",
    "trade_date": "",
    "start_date": 20201231,
    "end_date": 20221021,
    "limit": "",
    "offset": ""
}, fields=[
    "ts_code",
    "trade_date",
    "total_mv",

])
print(df)
# with open('./data/csi300_market_value_07to20.pkl', "rb") as fh:
#     df_market_value = pickle.load(fh)
# print(df_market_value/1000000000)