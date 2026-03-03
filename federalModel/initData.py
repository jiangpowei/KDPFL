import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


# 1. 加载数据
def load_data(file_path):
    df = pd.read_excel(file_path)
    # 移除不需要的列
    df = df.drop(['Date'], axis=1)

    return df


# 2. 创建时序数据集
import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length=30):
        # 假设 data 是一个二维 numpy 数组，最后一列是目标值
        self.X = data[:, :-1]  # 特征
        self.y = data[:, -1]   # 目标值
        self.seq_length = seq_length

    def __len__(self):
        return len(self.X) - self.seq_length

    def __getitem__(self, idx):
        # 返回一个序列和对应的目标值
        x = self.X[idx:idx + self.seq_length]
        y = self.y[idx + self.seq_length]
        return torch.FloatTensor(x), torch.FloatTensor([y])

# 3. 数据预处理和加载
def prepare_data(df, target_col='count', seq_length=30, test_size=0.2):
    # 分离特征和标签
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values

    # 数据标准化
    # scaler_X = StandardScaler()
    # scaler_y = StandardScaler()

    # X = scaler_X.fit_transform(X)
    # y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # 创建时序数据集
    dataset = TimeSeriesDataset(X, y, seq_length)

    # 划分训练集和测试集
    train_size = int(len(dataset) * (1 - test_size))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return (train_loader, test_loader  # , scaler_y
            )
