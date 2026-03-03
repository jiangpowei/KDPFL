import torch

def calculate_mae(predictions, targets):
    """计算平均绝对误差"""
    return torch.mean(torch.abs(predictions - targets)).item()

def calculate_rmse(predictions, targets):
    """计算均方根误差"""
    return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()

def calculate_mape(predictions, targets):
    """计算平均绝对百分比误差"""
    # 避免除零错误
    mask = targets > 1e-8  # 忽略接近零的目标值
    if mask.sum() == 0:
        return float('inf')
    return torch.mean(torch.abs((predictions[mask] - targets[mask]) / targets[mask])).item() * 100