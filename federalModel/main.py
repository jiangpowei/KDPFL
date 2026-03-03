import logging
import os

import flwr
import flwr as fl
from typing import List, Dict

import ray
import torch

from src.federalModel.TCN_client import TCNClient
from src.federalModel.configLogging import configure_logging
from src.federalModel.dataLoader import load_client_data
from src.federalModel.save_final_model import save_final_model_from_history
from src.federalModel.server_strategy import TCNStrategy, get_aggregated_parameters_history

# from data.data_loader import load_client_data
# from clients.tcn_client import TCNClient
# from server.server_strategy import TCNStrategy

import torch
print(f"PyTorch version: {torch.__version__}")

# 可选：同时查看是否支持CUDA（GPU加速）
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")


# 配置参数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LENGTH = 30  # 时序序列长度
NUM_CHANNELS = [64, 64, 64]  # TCN通道数
INPUT_SIZE = 20  # 输入特征数（根据实际数据调整）
OUTPUT_SIZE = 1  # 输出目标数
EPOCHS = 10  # 联邦学习轮数
CLIENT_IDS = [0, 1, 2]  # 客户端ID
DATA_PATHS = [
    r"C:\Users\jiang\Desktop\FL\data\2_AfterPreprocessing\CA_Fresno_93701.xlsx",  # 客户端1数据路径
    r"C:\Users\jiang\Desktop\FL\data\2_AfterPreprocessing\TX_Austin_78717.xlsx",  # 客户端2数据路径
    r"C:\Users\jiang\Desktop\FL\data\2_AfterPreprocessing\WI_Wausau_54401.xlsx"  # 客户端3数据路径
]

# 配置日志（在导入Flower和Ray之前调用）
configure_logging(suppress_all=False)  # 设置为True可完全禁用所有输出


def main():
    print(f"联邦学习将在 {DEVICE} 上运行")
    print(f"数据路径: {DATA_PATHS}")

    # 创建客户端
    clients = []
    for client_id, data_path in zip(CLIENT_IDS, DATA_PATHS):
        print(f"加载客户端 {client_id} 的数据: {data_path}")
        train_dataset, test_dataset, _, _ = load_client_data(data_path, SEQ_LENGTH)

        client = TCNClient(
            client_id=client_id,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            input_size=INPUT_SIZE,
            output_size=OUTPUT_SIZE,
            num_channels=NUM_CHANNELS
        )
        clients.append(client)

    # 定义联邦学习策略
    strategy = TCNStrategy(
        fraction_fit=1.0,  # 使用所有客户端进行训练
        fraction_evaluate=1.0,  # 使用所有客户端进行评估
        min_fit_clients=len(CLIENT_IDS),  # 至少需要所有客户端参与训练
        min_evaluate_clients=len(CLIENT_IDS),  # 至少需要所有客户端参与评估
        min_available_clients=len(CLIENT_IDS),  # 总客户端数量
    )

    # 启动联邦学习模拟
    print(f"开始 {EPOCHS} 轮的联邦学习训练...")
    fl.simulation.start_simulation(
        client_fn=lambda cid: clients[int(cid)],
        num_clients=len(CLIENT_IDS),
        config=fl.server.ServerConfig(num_rounds=EPOCHS),
        strategy=strategy,
        # 强制为每个客户端分配GPU资源（1块GPU分给3个客户端，每个0.33个GPU）
        client_resources={"num_cpus": 1, "num_gpus": 1.0 / len(CLIENT_IDS)},
    )

    # 获取手动保存的聚合参数
    aggregated_parameters_history = get_aggregated_parameters_history()
    if aggregated_parameters_history:
        # 修改这里，使用 parameters.tensors 访问参数
        final_parameters = aggregated_parameters_history[-1].tensors

        # 创建一个模拟的history对象
        class MockHistory:
            def __init__(self, parameters):
                self.parameters_aggregated = [parameters]

        mock_history = MockHistory(final_parameters)
        save_final_model_from_history(
            history=mock_history,
            input_size=INPUT_SIZE,
            output_size=OUTPUT_SIZE,
            num_channels=NUM_CHANNELS
        )
    else:
        print("未找到聚合参数，无法保存模型。")


if __name__ == "__main__":
    main()
