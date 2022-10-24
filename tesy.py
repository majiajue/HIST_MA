import pandas as pd
import pickle
import tushare as ts
import datetime as dt
import time
import numpy as np
# def func(x):
#     a=x
#     b = str(x)
#     a=a[7:9]+b[0:6]
#     return a
# pro = ts.pro_api('7bee558347c0922dfd601254294a5726654493fdf5545d5c28ee2153')
# # 拉取数据
# df = pro.trade_cal(**{
#     "exchange": "SSE",
#     "cal_date": "",
#     "start_date": 20210101,
#     "end_date": 20221021,
#     "is_open": "1",
#     "limit": "",
#     "offset": ""
# }, fields=[
#     "exchange",
#     "cal_date",
#     "is_open",
#     "pretrade_date"
# ])
# final_df = pd.DataFrame(columns=['datetime','instrument','$market_value'])
# for index, row in df.iterrows():
#     time.sleep(3)
#     value_df = pro.daily_basic(**{
#     "ts_code": "",
#     "trade_date": "",
#     "start_date": row['cal_date'],
#     "end_date": row['cal_date'],
#     "limit": "",
#     "offset": ""
# }, fields=[
#     "trade_date",
#     "ts_code",
#     "total_mv",
# ])
#     # value_df["trade_date"] = pd.to_datetime(value_df["trade_date"])
#     value_df["trade_date"]=value_df["trade_date"].apply(lambda x:dt.datetime.strptime(x,'%Y%m%d'))
#     value_df['ts_code']=value_df['ts_code'].apply(func)
#     value_df = value_df.rename(columns={'trade_date':'datetime','ts_code':'instrument','total_mv':'$market_value'})
#     value_df = value_df[['datetime','instrument','$market_value']]
#     final_df  = final_df .append(value_df,ignore_index=True)
# final_df.to_csv('value_df.csv',mode='a',encoding='utf-8',index=False)



# 初始化pro接口
# pro = ts.pro_api('7bee558347c0922dfd601254294a5726654493fdf5545d5c28ee2153')
# df = pro.daily_basic(**{
#     "ts_code": "",
#     "trade_date": "",
#     "start_date": 20201231,
#     "end_date": 20221021,
#     "limit": "",
#     "offset": ""
# }, fields=[
#     "ts_code",
#     "trade_date",
#     "total_mv",

# ])
# print(df)
# with open('./data/csi300_market_value_07to20.pkl', "rb") as fh:
#     df = pickle.load(fh)
# data=pd.read_pickle('./data/csi300_market_value_07to20.pkl')

# data.to_csv('df_market_value.csv',encoding='utf-8')
# data = pd.read_csv('df_market_value.csv',encoding='utf-8')
# # df.to_csv('df_market_value.csv',mode='a',encoding='utf-8',index=False)
# df = pd.read_csv('value_df.csv',encoding='utf-8')
# new_df = pd.concat([data,df],ignore_index=True)
# df2 = new_df.set_index(["datetime","instrument"])
# df2.to_pickle('./data/csi300_market_value_07to22.pkl')
# 导入tushare
import tushare as ts
# 初始化pro接口
pro = ts.pro_api('7bee558347c0922dfd601254294a5726654493fdf5545d5c28ee2153')

# 拉取数据
df = pro.fina_mainbz(**{
    "ts_code": "600001.SH",
    "period": "20090630",
    "type": "P",
    "end_date": "",
    "is_publish": "",
    "limit": "",
    "offset": ""
}, fields=[
    "ts_code",
    "bz_item",
    "end_date"

])
df2 = df.set_index(["end_date","ts_code"])["bz_item"]
df2  =  df2.drop_duplicates()
# new_df = df2.rename_axis(columns=None)
# new_df = new_df.reset_index()
print(df2)

# dd = df2.index.get_level_values(0).unique()
# assert len(dd == 1), '提取的stock2concept_matrix涉及多个交易日！'
# stock_code = df2.index.get_level_values(1).unique().tolist()
# print(dd[0])
stock2concept_matrix = np.load('./data/csi300_stock2concept.npy') 
print(len(stock2concept_matrix))
# def get_stock2concept_matrix_dt(stock2concept_matrix, index):
#     '''
#     按索引提取预定义概念
#     :param stock2concept_matrix:
#     :param index: 每个交易日下的index
#     :return:
#     '''
#     dt = index.get_level_values(0).unique()
#     assert len(dt == 1), '提取的stock2concept_matrix涉及多个交易日！'
#     stock_code = index.get_level_values(1).unique().tolist()
#     stock2concept_matrix_dt = stock2concept_matrix.loc[dt[0]].unstack().reindex(stock_code).fillna(0)
#     return stock2concept_matrix_dt
# stock_index = np.load('./data/csi300_stock_index.npy', allow_pickle=True).item()
# print(stock_index)

# stock2concept_matrix = np.load('./data/csi300_stock2concept.npy')
# print(stock2concept_matrix[1])