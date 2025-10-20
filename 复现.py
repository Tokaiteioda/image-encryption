import numpy as np
import cv2
import hashlib



def preprocessing(picture_path):  # 图像预检验
    img = cv2.imread(picture_path)
    height, width = img.shape[:2]  # 高度,长度
    if height != width:
        print("长宽不相等")


def key_generation(picture_path, password):  # 密钥生成
    """

    :param picture_path: 图片路径
    :param password: 口令
    :return: 归一化后的初始输入
    """
    with open(picture_path, 'rb') as f:
        data = f.read()
    hash_value = hashlib.sha256(data).digest()
    password_value = hashlib.sha256(password.encode()).digest()  # 字符串在哈希前必须先编码
    xor_hash = bytes(h ^ p for h, p in zip(hash_value, password_value))  # 异或操作

    x_bytes = xor_hash[0:11]
    y_bytes = xor_hash[11:22]
    z_bytes = xor_hash[22:32]

    def normalized(b):  # 归一化
        val_int = int.from_bytes(b, 'big')
        val_max = 2 ** (8 * len(b)) - 1
        return val_int / val_max

    x0 = normalized(x_bytes)
    y0 = normalized(y_bytes)
    z0 = normalized(z_bytes)

    return [x0, y0, z0]


def lorenz_system(state, a=10, b=8 / 3, c=28):  # Lorenz混沌序列
    x, y, z = state
    dx = a * (y - x)
    dy = x * (c - z) - y
    dz = x * y - b * z
    return np.array([dx, dy, dz])


def runge_kutta4(system, initial_state, h, steps):  # 四阶Runge-Kutta
    """
    :param system: 混沌系统
    :param initial_state: 初始状态
    :param h: 步长
    :param steps: 迭代次数
    :return:三条归一化后的混沌轨迹
    """
    state = np.array(initial_state, dtype=float)
    traj = np.zeros((steps, 3))  # 记录
    for i in range(steps):
        traj[i] = state
        k1 = h * system(state)
        k2 = h * system(state + 0.5 * k1)
        k3 = h * system(state + 0.5 * k2)
        k4 = h * system(state + k3)
        state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def normalize(arr):  # 归一化
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    x_norm = normalize(traj[:, 0])
    y_norm = normalize(traj[:, 1])
    z_norm = normalize(traj[:, 2])

    return np.array([x_norm, y_norm, z_norm])


picturePath = r"C:\Users\ltteo\Desktop\picture.jpg"
preprocessing(picturePath)
state = key_generation(picturePath, "hajimi")
print(state)
data = runge_kutta4(lambda s: lorenz_system(s, a=10, b=8 / 3, c=28), state, 0.01, 10000)
print(len(data[0]))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda")
data = data.T
sequence_length = 50 # 每50步预测下一步
X, Y = [], []
for i in range(len(data) - sequence_length):
    X.append(data[i: i + sequence_length])
    Y.append(data[i + sequence_length])
X = np.array(X)
Y = np.array(Y)
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)

dataset = TensorDataset(X_tensor, Y_tensor) # 小批量导入数据防止显存不足
loader = DataLoader(dataset, batch_size=64, shuffle=True)

class GRUChaos(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

model = GRUChaos().to(device)
criterion = nn.MSELoss() # 损失函数为均方误差
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---------- 训练 ----------
epochs = 300
for epoch in range(epochs):
    model.train()
    for X_batch, Y_batch in loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, Y_batch)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# ---------- 生成新的混沌序列 ----------
model.eval() # 停止训练
generated = []
input_seq = X_tensor[0].unsqueeze(0)  # 取第一段作为起点
steps_to_generate = 1000

for _ in range(steps_to_generate):
    with torch.no_grad():
        pred = model(input_seq)
    generated.append(pred.detach().cpu().numpy().flatten())
    # 滑动窗口更新输入
    new_input = torch.cat((input_seq[:, 1:, :], pred.unsqueeze(1)), dim=1)
    input_seq = new_input

generated = np.array(generated)  # shape (1000, 3)

print("新混沌序列维度:", generated.shape)