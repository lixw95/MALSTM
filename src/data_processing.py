import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import os
import requests, json
import sys
from fancyimpute import KNN
from jqdatasdk import *
import sqlalchemy
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import numpy
import torch
def data_processing(year):
    main_path = r"D:/download_code/DA-RNN-attchange1/DA-RNN-master/train_set220"
    file_name = '{}.csv'.format(year)
    file_path = os.path.join(main_path, file_name)
    df = pd.read_csv(file_path)
    x = df.loc[:, [x for x in df.columns.tolist() if x != 'code' and x != 'date']]
    if x.capitalization.isnull().any() or x.market_cap.isnull().any() or x.circulating_market_cap.isnull().any() or x.pe_ratio.isnull().any()\
        or x.pe_ratio_lyr.isnull().any() or x.pb_ratio.isnull().any() or x.total_assets.isnull().any() or x.total_liability.isnull().any()\
            or x.total_owner_equities.isnull().any() or x.operating_cost.isnull().any() or x.total_operating_cost.isnull().any()\
            or x.subtotal_operate_cash_inflow.isnull().any() or x.net_operate_cash_flow.isnull().any() or x.net_invest_cash_flow.isnull().any()\
            or x.cash_equivalent_increase.isnull().any():
        filled_knn = KNN(k=10).fit_transform(x)
        x = pd.DataFrame(filled_knn)
    x.to_csv(file_path)



if __name__ == '__main__':
    '''
    for year in range(1, 220):
        main_path = r"D:/download_code/DA-RNN-attchange1/DA-RNN-master/train_set_knn"
        file_name = '{}.csv'.format(year)
        file_path = os.path.join(main_path, file_name)
        df = pd.read_csv(file_path)
        row = df.shape[0]
        if row < 52:
            n = 52 - row
            fill = np.zeros((n, 17))
            cating = np.vstack((fill, df))
            dataframe = pd.DataFrame(cating)
            dataframe.to_csv(file_path, index=False, sep=',')
        else:
            year = year + 1
    '''
    for year in range(1, 220):
        main_path = r"D:/download_code/DA-RNN-attchange1/DA-RNN-master/train_set_knn"
        file_name = '{}.csv'.format(year)
        file_path = os.path.join(main_path, file_name)
        df = pd.read_csv(file_path)
        row = df.shape[0]
        if row < 52:
            print(year)
        else:
            year += 1
    print('完成！')
