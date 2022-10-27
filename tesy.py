import pandas as pd
import pickle
import tushare as ts
import datetime as dt
import time
import numpy as np
np.set_printoptions(threshold=np.inf)  
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

# stock_index = np.load('./data/csi300_stock_index.npy',allow_pickle=True).item()
# result_df = pd.DataFrame()
# for i in stock_index:
# # 拉取数据
#     time.sleep(2)
   
#     df = pro.fina_mainbz(**{
#         "ts_code":i[2:8]+"."+i[0:2],
#         "period": "",
#         "type": "P",
#         "end_date": "",
#         "is_publish": "",
#         "limit": "",
#         "offset": ""
#     }, fields=[
#         "ts_code",
#         "bz_item",
#         "end_date"

#     ])
#     result_df=result_df.append(df)
#     print(result_df)
    


# result_df.to_csv('stock_index2concept.csv',encoding='utf-8',index=False)
# df2 = df.set_index(["end_date","ts_code"])["bz_item"]
# df2  =  df2.drop_duplicates()
# new_df = df2.rename_axis(columns=None)
# new_df = new_df.reset_index()
# print(df2)
def getStock2concept():
    stock2concept = pd.read_csv('stock_index2concept.csv',encoding='utf-8')
    stock2concept['concept'] = 1
    df3 = stock2concept.set_index(["end_date","ts_code","bz_item"])
    df3 = df3[~df3.index.duplicated()]
    dt = df3.index.get_level_values(0).unique()
    stock_code = df3.index.get_level_values(1).unique().tolist()
    stock2concept_matrix_dt = df3.loc[dt[0]].unstack().reindex(stock_code).fillna(0)
    return stock2concept
# dt = df3.index.get_level_values(0).unique()
# df4=df3.loc[dt[0]].reset_index()
# print(df4)
# N, K = df4.shape
# relation = np.zeros((N, N, K-1))
# print(relation)
# for i in range(N):
#     for j in range(N):
#         for k in range(K-1):
#             if df4.iloc[i, k+1]==df4.iloc[j, k+1]:
#                 relation[i, j, k] = 1
# print(relation)
# print(len(stock2concept))
# df=stock2concept.drop_duplicates(subset=['end_date','ts_code','bz_item'], keep='first')
# print(df.head())
# df2 = stock2concept.value_counts(subset=['ts_code','end_date','bz_item'], sort=False).loc[lambda x: x > 1].to_frame().reset_index()
# df2 = df2.drop(columns=0)
# print(df2)
# df3 = df2.set_index(["ts_code","end_date"])
# df3 = df3[~df3.index.duplicated()]
# counts.to_csv('count.csv',encoding='utf-8')
# del stock2concept['index']
# df=stock2concept.drop_duplicates()
# df2 = df.groupby(by=["end_date","ts_code"])
# print(df.head())
# dt = df3.index.get_level_values(0).unique()
# df3 =df.loc[dt[0]]
# stock_code = df3.index.get_level_values(1).unique().tolist()
# stock2concept_matrix_dt = df3.loc[dt[0]].reindex(stock_code).fillna(0)
# print(stock2concept_matrix_dt)
# print(ddd)
# counts = df.value_counts(subset=['ts_code','end_date','bz_item'], sort=False).loc[lambda x: x > 1]
# print(counts)
# df2 = df.set_index(["end_date","ts_code"])
# print(df2)
# # stock2concept = stock2concept.set_index(["end_date","ts_code"])
# dt = df2.index.get_level_values(0).unique()
# # assert len(dt == 1), '提取的stock2concept_matrix涉及多个交易日！'
# stock_code = df2.index.get_level_values(1).unique().tolist()
# stock2concept_matrix_dt = df2.loc[dt[0]].drop_duplicates().unstack().reindex(stock_code)
# print(stock2concept_matrix_dt)
# print(stock2concept_matrix_dt)
# print(df2)
# df2 = pd.DataFrame(stock2concept,columns=["ts_code","bz_item"])
# df2 = df2.drop_duplicates()
# df2 = df2.set_index(["ts_code"])["bz_item"].unstack()
# print(df2)
# assert len(df2.index == 1)
# print(df2)
# dd = df2.index.get_level_values(0).unique()
# assert len(dd == 1), '提取的stock2concept_matrix涉及多个交易日！'
# stock_code = df2.index.get_level_values(1).unique().tolist()
# print(dd[0])
# stock2concept_matrix = np.load('./data/csi300_stock2concept.npy') 
# print(len(stock2concept_matrix[0]))
# print(len(stock2concept_matrix))
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