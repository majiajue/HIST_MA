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
import pickle as pickle
from model import MLP, HIST
import collections
from tqdm import tqdm
from utils import metric_fn, mse
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def retrain_model(model_path = "trained_model", fit_end_time="2022-08-01"):
    market = "csi300"
    benchmark = "SH000300"

    data_handler_config = {
        "start_time": "2006-01-01",
        "end_time": fit_end_time,
        "fit_start_time": "2008-01-01",
        "fit_end_time": fit_end_time,
        "instruments": "csi300",
        "benchmark": "SH000300",
        "days_ahead": 3
    }

    hanlder = {'class': 'Alpha360', 'module_path': 'qlib.contrib.data.handler', 'kwargs': {'start_time':  "2006-01-01", 'end_time': fit_end_time, 'fit_start_time':"2008-01-01", 'fit_end_time': fit_end_time, 'instruments': "csi300", 'infer_processors': [{'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'feature', 'clip_outlier': True}}, {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}], 'learn_processors': [{'class': 'DropnaLabel'}, {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}}], 'label': ['Ref($close, -1) / $close - 1']}}

    segments =  { 'train': (data_handler_config['fit_start_time'],data_handler_config['fit_end_time']), "valid": ("2005-08-01", "2007-12-31")}

    dataset = DatasetH(hanlder,segments)

    df_train, df_valid= dataset.prepare( ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L,)
    with open('./data/csi300_market_value_07to20.pkl', "rb") as fh:
        df_market_value = pickle.load(fh)
    df_market_value = df_market_value/1000000000
    stock_index = np.load('./data/csi300_stock_index.npy', allow_pickle=True).item()

    start_index = 0
    slc = slice(pd.Timestamp(data_handler_config['fit_start_time']), pd.Timestamp(data_handler_config['fit_end_time']))
    df_train['market_value'] = df_market_value[slc]
    df_train['market_value'] = df_train['market_value'].fillna(df_train['market_value'].mean())
    df_train['stock_index'] = 733
    df_train['stock_index'] = df_train.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)

    train_loader = DataLoader(df_train["feature"], df_train["label"], df_train['market_value'], df_train['stock_index'], batch_size=-1, pin_memory=True, start_index=start_index, device = device)

    slc = slice(pd.Timestamp("2005-08-01"), pd.Timestamp("2007-12-31"))
    df_valid['market_value'] = df_market_value[slc]
    df_valid['market_value'] = df_valid['market_value'].fillna(df_train['market_value'].mean())
    df_valid['stock_index'] = 733
    df_valid['stock_index'] = df_valid.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
    start_index += len(df_valid.groupby(level=0).size())

    valid_loader = DataLoader(df_valid["feature"], df_valid["label"], df_valid['market_value'], df_valid['stock_index'], pin_memory=True, start_index=start_index, device = device)

    stock2concept_matrix = np.load('./data/csi300_stock2concept.npy')
    stock2concept_matrix = torch.Tensor(stock2concept_matrix).to(device)
    recorder_name = "retrain_model"
    with R.start(recorder_name=recorder_name, experiment_name="train_model"):
        # model initiaiton
        model = HIST(d_feat = 6, num_layers = 2)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=2e-4)
        best_score = -np.inf
        best_epoch = 0
        stop_round = 0
        best_param = copy.deepcopy(model.state_dict())
        params_list = collections.deque(maxlen=5)
        model.train()
        global_step=-1
        for i, slc in tqdm(train_loader.iter_batch(), total=train_loader.batch_length):
            global_step += 1
            feature, label, market_value , stock_index, _ = train_loader.get(slc)
            pred = model(feature, stock2concept_matrix[stock_index], market_value)
            loss = loss_fn(pred, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
            optimizer.step()
        R.save_objects(trained_model=model)

        with Path(model_path).open("wb") as f:
            print("Saving model to", model_path)
            pickle.dump(model, f, protocol=4)
            print("Complete saving model")

def retrain_model_using_all_data(model_path = "trained_model"):
    provider_uri = "E://qlib//qlib_data//cn_data"  # target_dir
    if not exists_qlib_data(provider_uri):
        print(f"Qlib data is not found in {provider_uri}")
        return
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    retrain_model(model_path)

def loss_fn(pred, label):
    mask = ~torch.isnan(label)
    return mse(pred[mask], label[mask])
if __name__ == "__main__":
    fire.Fire(retrain_model_using_all_data)