import torch
import pandas as pd
from src.federalModel.TCN import TCN

# 配置参数，确保与训练时一致
INPUT_SIZE = 20
OUTPUT_SIZE = 1
NUM_CHANNELS = [64, 64, 64]
KERNEL_SIZE = 3
DROPOUT = 0.2
SEQUENCE_LENGTH = 30  # 固定时序长度

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
model.eval()  # 设置模型为评估模式

# 模拟样本数据，这里假设数据有 10 个样本，每个样本有 19 个特征
# 你需要将这部分替换为实际的数据读取和预处理操作
data = pd.read_excel(r"C:\Users\jiang\Desktop\FL\data\2_AfterPreprocessing\CA_Fresno_93701.xlsx")
data = data.drop(columns=['Date', 'count'], errors='ignore')

# 3. 检查特征列数是否为20
if data.shape[1] != INPUT_SIZE:
    raise ValueError(f"特征列数应为{INPUT_SIZE}，但实际为{data.shape[1]}")

# 4. 按每30行为一组处理数据（不足则零填充）
num_samples = len(data) // SEQUENCE_LENGTH  # 完整的30行样本数
remaining_rows = len(data) % SEQUENCE_LENGTH  # 剩余不足30行的数据

# 处理完整的30行样本
for i in range(num_samples):
    sample_data = data.iloc[i * SEQUENCE_LENGTH:(i + 1) * SEQUENCE_LENGTH, :].values
    sample_tensor = torch.tensor(sample_data, dtype=torch.float32).transpose(0, 1).unsqueeze(0)

    with torch.no_grad():
        prediction = model(sample_tensor)
        print(f"样本 {i + 1}（完整30行）预测结果：{prediction.item()}")

# 处理剩余不足30行的数据（零填充）
if remaining_rows > 0:
    # 创建零矩阵，形状为 (30 - 剩余行数, 20)
    padding = torch.zeros(SEQUENCE_LENGTH - remaining_rows, INPUT_SIZE)
    # 获取剩余数据并转为张量
    remaining_data = torch.tensor(data.iloc[-remaining_rows:, :].values, dtype=torch.float32)
    # 拼接零填充部分和剩余数据
    padded_data = torch.cat([padding, remaining_data], dim=0)  # 形状：(30, 20)
    # 调整维度为 [1, 20, 30]
    padded_tensor = padded_data.transpose(0, 1).unsqueeze(0)

    with torch.no_grad():
        prediction = model(padded_tensor)
        print(f"样本 {num_samples + 1}（零填充）预测结果：{prediction.item()}")