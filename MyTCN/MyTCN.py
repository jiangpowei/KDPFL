import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pytorch_tcn import TCN
import pandas as pd
import numpy as np
import matplotlib

# 依次尝试可用的后端，直到找到兼容的
for backend in ['TkAgg', 'QtAgg', 'WXAgg', 'Agg']:
    try:
        matplotlib.use(backend, force=True)
        break
    except:
        continue
import matplotlib.pyplot as plt

# ========== 1. 读取并处理数据 ==========
df = pd.read_excel(r'C:\Users\jiang\Desktop\FL\data\2_AfterPreprocessing\Zmerged_file.xlsx')

# 特征列：去除 Date 和 count 后其余列
feature_cols = df.columns.drop(['Date', 'count']).tolist()
target_col = 'count'

# 提取数据
X = df[feature_cols].values  # shape: (样本数, 特征数=20)
y = df[target_col].values  # shape: (样本数,)

# 构造前30天序列预测第31天
seq_len = 90
X_seq = []
y_seq = []

for i in range(len(df) - seq_len):
    X_seq.append(X[i:i + seq_len])  # shape: (30, 20)
    y_seq.append(y[i + seq_len])  # shape: ()

X_seq = np.array(X_seq)  # shape: (样本数, 30, 20)
y_seq = np.array(y_seq)  # shape: (样本数,)

# 转为 PyTorch 张量
X_tensor = torch.tensor(X_seq, dtype=torch.float32)  # (batch, seq_len, input_size)
y_tensor = torch.tensor(y_seq, dtype=torch.float32)  # (batch,)

# 调整维度：TCN 要求 (batch, input_size, seq_len)
X_tensor = X_tensor.permute(0, 2, 1)  # -> (batch, 20, 30)

# 创建数据集与加载器
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# ========== 2. 定义 TCN 模型 ==========
class TCNWrapper(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.tcn = TCN(
            num_inputs=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.scale_factor = 500  # 缩放因子

    def custom_sigmoid(self, x):
        """自定义缩放Sigmoid函数: σ(x) = 500 * 1/(1+e^(-x/500))"""
        return self.scale_factor / (1 + torch.exp(-x / self.scale_factor))

    def forward(self, x):
        tcn_output = self.tcn(x)  # -> (batch, num_channels[-1], seq_len)
        last_output = tcn_output[:, :, -1]  # -> (batch, num_channels[-1])
        # 取最后一个通道的输出作为特征表示
        feature = last_output[:, -1].unsqueeze(1)  # -> (batch, 1)
        # 通过自定义sigmoid激活函数
        output = self.custom_sigmoid(feature)
        return output  # -> (batch, 1)


# ========== 3. 初始化模型参数 ==========
input_size = len(feature_cols)  # = 20
num_channels = [25, 25, 25]
kernel_size = 3
dropout = 0.2

# ========== 4. 配置GPU支持 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = TCNWrapper(
    input_size=input_size,
    num_channels=num_channels,
    kernel_size=kernel_size,
    dropout=dropout
).to(device)  # Move model to GPU

# ========== 5. 定义损失与优化器 ==========
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ========== 6. 开始训练并记录损失 ==========
num_epochs = 100
train_losses = []  # Record loss for each epoch

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        # Move data to GPU
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Calculate and record average loss
    epoch_loss = running_loss / len(dataloader)
    train_losses.append(epoch_loss)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# ========== 7. 保存模型 ==========
torch.save(model.state_dict(), 'tcn_model.pth')
print("Model saved as: tcn_model.pth")

# ========== 8. 绘制损失曲线 ==========
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, 'b-', linewidth=2)
plt.title('Training Loss Curve', fontsize=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('training_loss.png', dpi=300)
plt.show()