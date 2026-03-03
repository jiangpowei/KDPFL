import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import flwr as fl
from typing import List, Dict, Tuple, Optional


# TCN基础模块：残差块
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # 保存padding参数，用于后续截断
        self.padding = padding

        # 第一层因果卷积（仅左侧填充，右侧不填充）
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二层因果卷积
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 残差连接
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x形状：[batch_size, input_size, sequence_length]
        input_len = x.shape[2]  # 记录输入序列长度

        # 第一层卷积 + 截断右侧冗余（确保输出长度=输入长度）
        out = self.conv1(x)
        out = out[:, :, :input_len]  # 关键：截断到输入长度（删除右侧多余部分）
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        # 第二层卷积 + 截断右侧冗余
        out = self.conv2(out)
        out = out[:, :, :input_len]  # 再次截断到输入长度
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # 残差连接（res长度=输入长度，与out匹配）
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# 完整TCN模型
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1, 2, 4, ...
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            # 计算padding：必须满足 2×padding = dilation×(kernel_size-1)
            padding = (dilation_size * (kernel_size - 1)) // 2  # 核心修正
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=padding,  # 使用计算后的padding
                dropout=dropout
            )]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # 输入x形状: [batch_size, sequence_length, input_size]
        x = x.permute(0, 2, 1)  # 转换为 [batch_size, input_size, sequence_length]
        y1 = self.network(x)
        y1 = y1[:, :, -1]  # 取最后一个时间步的输出
        return self.linear(y1)