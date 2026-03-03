import torch
from src.federalModel.TCN import TCN

# 配置参数，确保与训练时一致
INPUT_SIZE = 20
OUTPUT_SIZE = 1
NUM_CHANNELS = [64, 64, 64]
KERNEL_SIZE = 3
DROPOUT = 0.2

# 创建模型实例
model = TCN(
    input_size=INPUT_SIZE,
    output_size=OUTPUT_SIZE,
    num_channels=NUM_CHANNELS,
    kernel_size=KERNEL_SIZE,
    dropout=DROPOUT
)

# 加载模型权重
model_path = r"C:\Users\jiang\Desktop\FederalLearning\src\federalModel\saved_models\federated_tcn_model.pth"
model.load_state_dict(torch.load(model_path))

# 输出模型各层的权重矩阵
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Layer name: {name}")
        print(f"Weight matrix shape: {param.data.shape}")
        print(f"Weight matrix values: \n{param.data}\n")