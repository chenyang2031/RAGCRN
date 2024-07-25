import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
from scipy import sparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(file_name):
    
    df = pd.read_csv(file_name, encoding='utf-8')
    df = df.to_numpy()
    MAX = np.max(df, axis=0, keepdims=True)
    MIN = np.min(df, axis=0, keepdims=True)
    df = (df - MIN) / (MAX - MIN)
    return df, MAX, MIN

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

def nn_seq(file_name, B):
    print('data processing...')
    data, MAX, MIN = load_data(file_name) 
    load = data
    load = torch.FloatTensor(load)
    seq = []
    for i in range(len(data)-12-3):
        train_seq = []
        train_label = []
        for m in range(i, i + 12):
            train_seq.append(load[m])
        for k in range(i+12, i + 12+3):
            train_label.append(load[k])
        train_seq = torch.stack(train_seq)
        train_label = torch.stack(train_label)
        train_seq = train_seq.transpose(1, 0)
        train_label = train_label.transpose(1, 0)
        seq.append((train_seq, train_label))
    Dtr = seq[0:int(len(seq) * 0.7)] 
    zhuxian=seq[int(len(seq) * 0.7):int(len(seq) * 0.8)]
    Dte = seq[int(len(seq) * 0.8):len(seq)] 
    train_len = int(len(Dtr) / B) * B  
    test_len = int(len(Dte) / B) * B  
    val_len=int(len(zhuxian) / B) * B
    Datatrain,  Dataval ,  Datatest = Dtr[:], zhuxian[:] , Dte[:]
    train = MyDataset(Datatrain)
    test = MyDataset(Datatest)
    val= MyDataset(Dataval)
    Datatrain = DataLoader(dataset=train, batch_size=B, shuffle=True, num_workers=0)
    Datatest = DataLoader(dataset=test, batch_size=B, shuffle=False, num_workers=0)
    Dataval = DataLoader(dataset=val, batch_size=B, shuffle=False, num_workers=0)
    return Datatrain,Dataval, Datatest, MAX, MIN

def mape_(x, y):
    return np.mean(np.abs((x - y) / (x + 0.001)))

def mae_(x, y):
    return np.mean(np.abs(x - y))

def rmse_(x, y):
    return np.sqrt(np.mean(np.power(x - y, 2)))

def r2_(x,y):

    return 1 - np.sum((x - y) ** 2) / np.sum((x - np.mean(x)) ** 2)

def total(x, y):
    mae = mae_(x, y)
    mape = mape_(x, y)
    rmse = rmse_(x, y)
    r2=r2_(x,y)

    return mae, mape, rmse,r2

if __name__ == '__main__':
    data = "../data/los_speed.csv"
    Datatrain, Datatest, MAX, MIN= nn_seq(data, 32)
