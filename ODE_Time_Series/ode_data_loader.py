import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class dataloader_ode(Dataset):
    """_summary_

    Args:
        Dataset (pandas DF): dataset containing lagged values of time series
            Example: using stock price with a 11 lag lookback with a 12 month forecast
                    creating a df with 24 columns. The first 12 includes the lags and
                    last 12 include the correct predictions.
                df[0, 0] = 115.0
                .
                .
                .
                df[0, 11] = 105.50 (the price 12 months ago)
                ------
                df[0, 12] = 117.0 (the price 1 month into the future)
                .
                .
                .
                df[0, 23] = 120.0 (the price 12 months into the future)
        scale_max: 4000 (arbitrary choice)
        scale_min: dataset min
        new_max: 1.0
        new_min: -1.0
    """
    def __init__(self, filepath, train=True):
        super(dataloader_ode,self).__init__()
        self.df = pd.read_csv(filepath)
        num_rows_train = round(.8*self.df.shape[0])

        if train == True:
            self.df = self.df.iloc[0:num_rows_train,:]
            
        self.scale_max = 4000.0
        self.scale_min = self.df.min().min()
        self.new_max = 1.0
        self.new_min = -1.0
        
    def __len__(self,idx):
        return self.df.shape[0]
    
    def __getitem__(self,idx):
        scaler_df = self.df.iloc[idx,:0:13]
        scaler_trgt = self.df.iloc[idx,13:]
        
        df_x = (scaler_df - self.scale_min)/(self.scale_max - self.scale_min)*(self.new_max-self.new_min) + self.new_min
        df_y = (scaler_trgt - self.scale_min)/(self.scale_max - self.scale_min)*(self.new_max-self.new_min) + self.new_min
        
        return torch.Tensor([df_x]).float(), torch.Tensor([df_y],torch.float).float()






























