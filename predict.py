import qlib
import torch
import torch.nn as nn
import torch.optim as optim
from qlib.workflow import R
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
import pickle
import os
import copy
from pathlib import Path
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
import numpy as np
import pandas as pd
import fire
from dataloader import DataLoader
import pickle
from model import MLP, HIST
import collections
from tqdm import tqdm
from utils import metric_fn, mse
import datetime
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def get_stock2concept_matrix_dt(stock2concept_matrix, index):
    '''
    按索引提取预定义概念
    :param stock2concept_matrix:
    :param index: 每个交易日下的index
    :return:
    '''
    dt = index.get_level_values(0).unique()
    assert len(dt == 1), '提取的stock2concept_matrix涉及多个交易日！'
    stock_code = index.get_level_values(1).unique().tolist()
    stock2concept_matrix_dt = stock2concept_matrix.loc[dt[0]].unstack().reindex(stock_code).fillna(0)
    return stock2concept_matrix_dt
def dump_predict_using_model(model_path = "trained_model", output_directory = "./"):
  # Init data
    provider_uri = "~//.qlib//qlib_data//cn_data"
    if not exists_qlib_data(provider_uri):
        raise Exception(f"Qlib data is not found in {provider_uri}")
    qlib.init(provider_uri=provider_uri, region=REG_CN)

  # Prepare dataset 

    hanlder = {'class': 'Alpha360', 'module_path': 'qlib.contrib.data.handler', 'kwargs': {'start_time':  "2006-01-01", 'end_time': "2055-07-22", 'fit_start_time':"2022-07-01", 'fit_end_time':"2022-07-02", 'instruments': "csi300", 'infer_processors': [{'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'feature', 'clip_outlier': True}}, {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}], 'learn_processors': [{'class': 'DropnaLabel'}, {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}}], 'label': ['Ref($close, -1) / $close - 1']}}

    segments =  {"test": ("2022-07-01", "2055-07-22")}
    with open('./data/csi300_market_value_07to22.pkl', "rb") as fh:
        df_market_value = pickle.load(fh)
    df_market_value = df_market_value/1000000000
    stock_index = np.load('./data/csi300_stock_index.npy', allow_pickle=True).item()
    start_index = 0
    df_market_value=df_market_value.reset_index()
    df_market_value['datetime'] = pd.to_datetime(df_market_value['datetime'],format='%Y-%m-%d')
    df_market_value.set_index(['datetime','instrument'], inplace=True)
    slc = slice(pd.Timestamp("2022-07-01"), pd.Timestamp("2055-07-22"))
    dd = df_market_value[slc]
    df_market_value=df_market_value[~df_market_value.index.duplicated()]
    print(type(dd))
    print(dd)
    dataset = DatasetH(hanlder,segments)
    df_test = dataset.prepare("test", data_key=DataHandlerLP.DK_L,)
    df_test = df_test[~df_market_value.index.duplicated()]
    print(df_test.head())
    # data=pd.DataFrame(np.array(df_test))
    print(type(df_test))
    # df_test2 = pd.merge(df_test, dd, left_index=True, right_index=True)
    # print(df_test2)
    # print(df_test2)
    df_test['market_value'] = dd
    # df_test['market_value'] = df_test['market_value'].fillna(df_test['market_value'].mean())
    # df_test['stock_index'] = 733
    # df_test['stock_index'] = df_test.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
    # start_index += len(df_test.groupby(level=0).size())
    # print(df_test)
    # test_loader = DataLoader(df_test["feature"], df_test["label"], df_test['market_value'], df_test['stock_index'], pin_memory=True, start_index=start_index, device = device)
    # stock2concept_matrix = np.load('./data/csi300_stock2concept.npy')
    # stock2concept_matrix = torch.Tensor(stock2concept_matrix).to(device)
    # preds = pd.DataFrame()
    # with Path(model_path).open("rb") as f:
    #     model = pickle.Unpickler(f).load()
    # model.eval()
    # for i, slc in tqdm(test_loader.iter_daily(), desc=stock2concept_matrix, total=test_loader.daily_length):

    #     feature, label, market_value, stock_index, index = test_loader.get(slc)

    #     pred = model(feature, stock2concept_matrix[stock_index], market_value)
    #     print(pred)
    #     preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy(), }, index=index))
    
    # pred_df = pd.concat(preds, axis=0)
    # print(preds)
    # last_item_index = pred_df.iloc[-1:].index[0]
    # last_item_datetime = last_item_index[0]
    # latest_score_list = pred_df.at[last_item_datetime].sort_values(ascending=False)

    # output_file_name = last_item_datetime.strftime("%Y-%m-%d") + ".csv"

    # output_file_path = os.path.join(output_directory, output_file_name)
    # latest_score_list.to_csv(output_file_path)

if __name__ == "__main__":
    fire.Fire(dump_predict_using_model)