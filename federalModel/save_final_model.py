import os
import torch
from src.federalModel.TCN import TCN

def save_final_model_from_history(history, input_size, output_size, num_channels,
                                  kernel_size=3, dropout=0.2, save_dir="saved_models"):
    """从模拟历史中保存最终模型"""

    # 从history中获取最终聚合的参数
    if not hasattr(history, "parameters_aggregated") or not history.parameters_aggregated:
        raise ValueError("无法从history中获取聚合参数，请确保Flower版本支持此功能")

    # 获取最后一轮的参数
    final_parameters = history.parameters_aggregated[-1]

    # 创建模型实例
    model = TCN(
        input_size=input_size,
        output_size=output_size,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout
    )

    # 将参数转换为模型权重
    params_dict = zip(model.state_dict().keys(), final_parameters)
    state_dict = {}
    for k, v in params_dict:
        try:
            # 尝试直接转换为torch.tensor
            state_dict[k] = torch.tensor(v)
        except TypeError:
            # 如果遇到bytes类型，尝试解码为合适的类型
            try:
                # 假设bytes类型的数据是numpy数组的二进制表示
                import numpy as np
                v_np = np.frombuffer(v, dtype=np.float32)
                state_dict[k] = torch.tensor(v_np)
            except Exception as e:
                print(f"处理参数 {k} 时出错: {e}")
                continue

    # 加载state_dict，使用strict=False忽略缺失键和形状不匹配问题
    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        print(f"加载state_dict时出现问题: {e}")

    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    # 保存模型权重
    model_path = os.path.join(save_dir, "federated_tcn_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"最终模型已保存到: {model_path}")

    return model_path