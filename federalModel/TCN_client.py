import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl
from typing import Dict, List, Optional, Tuple

from src.training.TCN import TCN


class TCNClient(fl.client.NumPyClient):
    def __init__(self, client_id, train_dataset, test_dataset, input_size, output_size, num_channels):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # 子进程中重新检测GPU（分配资源后应能检测到）
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # 打印设备信息，验证子进程是否正确识别GPU
        print(f"客户端 {client_id} 初始化设备: {self.device}, GPU数量: {torch.cuda.device_count()}")

        # 模型加载到当前设备（此时应为cuda）
        self.model = TCN(
            input_size=input_size,
            output_size=output_size,
            num_channels=num_channels,
            kernel_size=3,
            dropout=0.2
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.batch_size = 32

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    # 在 TCNClient 的 get_parameters 方法中（参数从GPU转到CPU）
    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> List[np.ndarray]:
        # 先将GPU上的张量移到CPU，再转为numpy数组（避免序列化CUDA对象）
        return [param.detach().cpu().numpy() for param in self.model.parameters()]

    # 在 fit 方法中接收参数时（从CPU转到GPU）
    def fit(self, parameters, config):
        for param, new_param in zip(self.model.parameters(), parameters):
            # 先将numpy数组转为CPU张量，再移到客户端的GPU
            param.data = torch.tensor(new_param).to(self.device)

        # 本地训练
        self.model.train()
        train_loss = 0.0
        train_mae = 0.0

        for batch_X, batch_y in self.train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device).squeeze()

            outputs = self.model(batch_X).squeeze()
            loss = self.criterion(outputs, batch_y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() * batch_X.size(0)
            train_mae += torch.sum(torch.abs(outputs - batch_y)).item()

        train_loss /= len(self.train_dataset)
        train_mae /= len(self.train_dataset)
        print(f"客户端 {self.client_id} 本地训练 - 损失: {train_loss:.4f}, MAE: {train_mae:.4f}")

        return self.get_parameters({}), len(self.train_dataset), {"loss": train_loss, "mae": train_mae}

    # 在 evaluate 方法中同样处理参数接收
    def evaluate(self, parameters, config):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param).to(self.device)

        # 评估
        self.model.eval()
        test_loss = 0.0
        test_mae = 0.0

        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device).squeeze()
                outputs = self.model(batch_X).squeeze()
                loss = self.criterion(outputs, batch_y)

                test_loss += loss.item() * batch_X.size(0)
                test_mae += torch.sum(torch.abs(outputs - batch_y)).item()

        test_loss /= len(self.test_dataset)
        test_mae /= len(self.test_dataset)
        print(f"客户端 {self.client_id} 本地评估 - 损失: {test_loss:.4f}, MAE: {test_mae:.4f}")

        return test_loss, len(self.test_dataset), {"loss": test_loss, "mae": test_mae}