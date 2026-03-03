from typing import Dict, List, Optional, Tuple, Union
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics

111111
# 全局变量用于保存每一轮的聚合参数
aggregated_parameters_history = []

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """聚合评估指标（加权平均）"""
    # 计算每个客户端的样本权重
    samples = [num_examples for num_examples, _ in metrics]
    total_samples = sum(samples)
    weights = [num_examples / total_samples for num_examples in samples]

    # 加权平均损失和MAE
    loss = sum([w * m["loss"] for w, (_, m) in zip(weights, metrics)])
    mae = sum([w * m["mae"] for w, (_, m) in zip(weights, metrics)])

    return {"loss": loss, "mae": mae}


class TCNStrategy(FedAvg):
    """自定义联邦学习策略"""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, Exception], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """聚合训练结果并保存参数"""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            aggregated_parameters_history.append(aggregated_parameters)
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
            failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, Exception], BaseException]],  # 修改这里
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        """聚合评估结果并打印"""
        if not results:
            return None, {}

        # 调用父类方法获取聚合损失
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # 自定义聚合指标
        metrics = [(r.num_examples, r.metrics) for _, r in results]
        aggregated_metrics = weighted_average(metrics)

        # 打印聚合结果
        print(
            f"第 {server_round} 轮联邦评估: 损失={aggregated_metrics['loss']:.4f}, MAE={aggregated_metrics['mae']:.4f}")

        return aggregated_loss, aggregated_metrics

def get_aggregated_parameters_history():
    return aggregated_parameters_history