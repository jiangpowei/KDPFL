import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.federalModel.initData import TimeSeriesDataset


def load_client_data(data_path, seq_length=30, test_size=0.2):
    # 读取数据（支持Excel/csv等，根据实际调整）
    if data_path.endswith('.xlsx') or data_path.endswith('.xls'):
        df = pd.read_excel(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    df = df.drop(['Date'], axis=1)
    # df = df.drop(['Maximum Wind Gust'], axis=1)


    # 提取特征和目标
    features = df.iloc[:, :-1].values
    target = df.iloc[:, -1].values

    # 合并数据
    data = np.column_stack((features, target))

    # 划分训练集和测试集
    train_size = int(len(data) * (1 - test_size))
    train_data = data[:train_size]
    test_data = data[train_size:]

    # 创建数据集
    train_dataset = TimeSeriesDataset(train_data, seq_length)
    test_dataset = TimeSeriesDataset(test_data, seq_length)

    return (train_dataset, test_dataset, None, None)