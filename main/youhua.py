# gru_image_encrypt_gpu.py
import os
import hashlib
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import time

# ---------------------------
# 固定随机种子，保证可复现
# ---------------------------
def set_seed(seed=12345):
    np.random.seed(seed)
    torch.manual_seed(seed)  # torch上CPU的随机种子
    if torch.cuda.is_available(): # torch上GPU的随机种子
        torch.cuda.manual_seed_all(seed)

# ---------------------------
# 从图片和口令生成初始密钥状态和 Arnold 迭代次数
# ---------------------------
def key_generation(picture_path, password):
    """返回初始状态 [x0, y0, z0] 和 Arnold 迭代次数"""
    with open(picture_path, 'rb') as f:
        data = f.read()
    hash_value = hashlib.sha256(data).digest()
    password_value = hashlib.sha256(password.encode()).digest()
    xor_hash = bytes(h ^ p for h, p in zip(hash_value, password_value))

    # 切分为三段，用于初始化 Lorenz 系统
    x_bytes = xor_hash[0:11]
    y_bytes = xor_hash[11:22]
    z_bytes = xor_hash[22:32]

    def normalized(b): # 归一化
        val_int = int.from_bytes(b, 'big') # bytes -> int : b[0] * (256 ^ 0) + b[1] * (256 ^ 1) + ...
        val_max = 2 ** (8 * len(b)) - 1 # max_bytes -> max_int : 2 ^ (len * 11) - 1
        return val_int / val_max

    x0 = normalized(x_bytes)
    y0 = normalized(y_bytes)
    z0 = normalized(z_bytes)

    # 用 hash 的前两个字节生成 Arnold 迭代次数
    iter_seed = xor_hash[0] + xor_hash[1] * 256 # b[0] + b[1] * 256
    arnold_iters = (iter_seed % 30) + 1

    return [x0, y0, z0], arnold_iters

# ---------------------------
# Lorenz 混沌系统（GPU 计算）
# ---------------------------
def lorenz_system_tensor(state, a=10, b=8/3, c=28):
    """输入 state: (3,) 张量, 输出 dx, dy, dz"""
    x, y, z = state           # Lorenz公式
    dx = a * (y - x)          # dx/dt = a * (y - x)
    dy = x * (c - z) - y      # dy/dt = x * (c - z) - y
    dz = x * y - b * z        # dz/dt = x * y - b * z
    return torch.stack([dx, dy, dz], dim=-1)  # .stack保留计算图

# ---------------------------
# GPU 版 RK4 积分生成混沌轨迹
# ---------------------------
def runge_kutta4_tensor(system, initial_state, h, steps, device=None):
    """

    Args:
        system: 微分方程函数
        initial_state: 初始状态[x0, y0, z0]
        h: 时间步长
        steps: 积分步数
        device: 计算设备

    Returns: 3维混沌序列

    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state = torch.tensor(initial_state, dtype=torch.float32, device=device)
    traj = torch.zeros((steps, 3), dtype=torch.float32, device=device) # 轨迹张量traj，记录(x, y, z)状态

    for i in tqdm(range(steps), desc="Lorenz RK4部分"):   # 逐步积分
        traj[i] = state                     # 保留当前状态
        k1 = h * system(state)              # k1: 当前斜率
        k2 = h * system(state + 0.5 * k1)   # k2: 当前点加半步 k1 的斜率
        k3 = h * system(state + 0.5 * k2)   # k3：当前点加半步 k2 的斜率
        k4 = h * system(state + k3)         # k4：当前点加一步 k3 的斜率
        state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0  # state_next = state + (k1 + 2k2 + 2k3 + k4) / 6

    # 每通道归一化到 [0,1]
    def normalize_col(col):
        mn = torch.min(col)
        mx = torch.max(col)
        if mx - mn == 0:
            return torch.zeros_like(col)
        return (col - mn) / (mx - mn)

    x_norm = normalize_col(traj[:, 0])
    y_norm = normalize_col(traj[:, 1])
    z_norm = normalize_col(traj[:, 2])
    return torch.stack([x_norm, y_norm, z_norm], dim=1)  # (steps,3)

# ---------------------------
# GRU 神经网络模型
# ---------------------------
class GRUChaos(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=3, output_size=3):
        """
        Args:
            input_size: 输入维度(3)
            hidden_size: 隐藏层维度
            num_layers: GRU层数(3)
            output_size: 输出维度
        """
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(  # 顺序执行
            nn.Linear(hidden_size, output_size), # 全连接层
            nn.Sigmoid()  # 激活函数输出映射到 [0,1]
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        # batch : 一次输入的序列数量 seq_len : 每个序列长度
        out, h = self.gru(x) # 获取输出序列提取时间特征
        # 仅取最后一个时间步作为最终状态输出
        out_last = out[:, -1, :]
        return self.fc(out_last), h

# ---------------------------
# 图像裁剪到中心正方形
# ---------------------------
def crop_center_square(img):
    h, w = img.shape[:2]
    if h == w:
        return img
    side = min(h, w)
    # 裁剪居中
    top = (h - side) // 2 # top : 从上向下裁剪的起点
    left = (w - side) // 2 # left : 从左往右裁剪的起点
    return img[top:top+side, left:left+side]

# ---------------------------
# GPU 版 Arnold 变换（单通道）
# ---------------------------
def arnold_transform_channel_tensor(channel, iterations=1, a=1, b=1, device=None):
    """输入 channel: (N,N) 张量, 返回置乱后的张量"""
    if device is None:
        device = channel.device

    N = channel.shape[0]
    X, Y = torch.meshgrid(                           # 像素位置网格坐标
        torch.arange(N, device=device),
        torch.arange(N, device=device),
        indexing='ij' # 高宽
    )

    res = channel.to(device).clone() # 初始通道的副本
    for _ in range(iterations): # [x_new, y_new].T = [[1, a],[b, ab+1]][x, y].T mod N
        # Arnold 矩阵 M = [[1, a],[b, ab+1]]
        X_new = (X + a * Y) % N
        Y_new = (b * X + (a * b + 1) * Y) % N
        new = torch.zeros_like(res)
        new[X_new, Y_new] = res[X, Y]
        res = new                    # 保存当前变换结果
    return res

def inverse_arnold_transform_channel_tensor(channel, iterations=5, a=1, b=1, device=None):
    """逆 Arnold 变换"""
    if device is None:
        device = channel.device

    N = channel.shape[0]
    X_scrambled, Y_scrambled = torch.meshgrid(        # 像素位置网格坐标
        torch.arange(N, device=device),
        torch.arange(N, device=device),
        indexing='ij'
    )

    # Arnold 逆矩阵 M_inv = [[ab+1, -a], [-b, 1]]
    a_inv = a * b + 1  # 2
    b_inv = -a         # -1
    c_inv = -b         # -1
    d_inv = 1          # 1

    res = channel.to(device).clone()
    for _ in range(iterations):   # [x, y].T = M_inv[x_new, y_new]
        X_plain = (a_inv * X_scrambled + b_inv * Y_scrambled) % N
        Y_plain = (c_inv * X_scrambled + d_inv * Y_scrambled) % N
        new = torch.zeros_like(res)
        new[X_plain, Y_plain] = res[X_scrambled, Y_scrambled] # 像素转移
        res = new

    return res

# ---------------------------
# 将 GRU 输出浮点序列映射到 uint8 密钥字节
# ---------------------------
def seq_to_key_bytes_tensor(generated_seq, num_bytes=1):
    """输入 generated_seq: (N,3) 张量,每一层为三维浮点向量(x, y ,z),
       返回 kr, kg, kb"""
    seq = torch.clamp(generated_seq, 0.0, 1.0)  # 保证序列元素在[0, 1]内, 小于0.0则置为0.0, 大于1.0则置为1.0
    # 组合不同通道值增加混沌度, % 1.0 保证在[0, 1]
    combo_r = (seq[:, 0] + seq[:, 1]) % 1.0  # R通道 : R + G
    combo_g = (seq[:, 1] + seq[:, 2]) % 1.0  # G通道 : G + B
    combo_b = (seq[:, 2] + seq[:, 0]) % 1.0  # B通道 : B + R

    def to_bytes(arr):  # 浮点映射为字节
        val = torch.floor((arr * 1e6) % 256).to(torch.uint8) # 限制在[0, 255]内
        return val

    # 生成三通道密钥
    kr = to_bytes(combo_r)
    kg = to_bytes(combo_g)
    kb = to_bytes(combo_b)
    return kr, kg, kb

# ---------------------------
# Chaotic Diffusion（加密/解密）
# ---------------------------
def chaotic_diffusion_rgb_tensor(img_rgb, kr, kg, kb, device=None):
    """输入 img_rgb: (H,W,3) uint8 张量, kr,kg,kb: 1D 密钥张量"""
    # 扩散公式 : Ci = (Pi + Ki + Ci-1) mod 256
    if device is None:
        device = img_rgb.device

    h, w, _ = img_rgb.shape
    flat = img_rgb.reshape(-1,3).to(torch.int16).to(device) # 将三维图像数组(H, W, 3)转为(H*W, 3)的二维数组
    k = torch.stack([kr, kg, kb], dim=1).to(torch.int16).to(device=device) # 合并密钥数组

    tmp = (flat + k) % 256 # 每个像素加上密钥
    cipher = torch.cumsum(tmp, dim=0) % 256 # 利用前缀和实现链式积累
    return cipher.reshape(h,w,3).to(torch.uint8) # 将二维数组(nums, 3)转为三维图像数组(H, W, 3)

def inverse_chaotic_diffusion_rgb_tensor(cipher_rgb, kr, kg, kb, device=None):
    # 逆扩散公式 : Pi = (Ci - Ki - Ci-1) mod 256
    if device is None:
        device = cipher_rgb.device

    h, w, _ = cipher_rgb.shape
    num = h * w
    flat = cipher_rgb.reshape(-1,3).to(torch.int16).to(device) # 将三维图像数组(H, W, 3)转为(num, 3)的二维数组，逐像素处理
    k = torch.stack([kr, kg, kb], dim=1).to(torch.int16).to(device) # 提前堆叠密钥张量

    # vectorized inverse: tmp[0]=cipher[0]; tmp[i]=cipher[i]-cipher[i-1]
    # 注意模 256 的正确处理（使用整数并加上256再模）
    prev = torch.zeros((1, 3), dtype=torch.int16, device=device)
    cipher_shifted = torch.cat([prev, flat[:-1]], dim=0)
    tmp = (flat - cipher_shifted) % 256
    plain = (tmp - k) % 256
    return plain.to(torch.uint8).reshape(h, w, 3)
# ---------------------------
# 预处理图像，BGR -> RGB + 裁剪正方形
# ---------------------------
def prepare_image_for_arnold(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    square = crop_center_square(rgb)
    return square

# ---------------------------
# 图像加密主流程
# ---------------------------
def encrypt_image(input_path, password, model_path=None, save_model_path='gru_chaos_checkpoint.pth',
                  train_if_no_model=False, rk_steps=20000, rk_dt=0.01, seq_len=50):
    """
    Args:
        input_path: 图像路径
        password: 用户口令
        model_path: 指定模型路径
        save_model_path: 模型存储路径
        train_if_no_model: 是否在无模型时训练新的GRU
        rk_steps: 积分步数
        rk_dt: 积分步长
        seq_len: 训练输入序列长度

    Returns: encrypted_rgb.cpu().numpy():最终加密图像
             metadata：加密相关信息
             model：GRU 模型对象
    """
    set_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取图像
    img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图片: {input_path}")
    img_rgb = prepare_image_for_arnold(img_bgr)
    h, w = img_rgb.shape[:2]
    num_pixels = h * w

    # 生成初始密钥[x0, y0, z0]和 Arnold 迭代次数
    init_state, arnold_iters = key_generation(input_path, password)

    # Lorenz + RK4 生成训练序列
    lorenz_steps = max(rk_steps, num_pixels + seq_len + 100)
    lorenz_seq = runge_kutta4_tensor(lorenz_system_tensor, init_state, rk_dt, lorenz_steps, device=device)

    # 构建 GRU 数据集
    X, Y = [], []
    for i in tqdm(range(len(lorenz_seq) - seq_len), desc="加密GRU数据集构建部分"):
        X.append(lorenz_seq[i:i+seq_len].cpu().numpy())  # 输入序列
        Y.append(lorenz_seq[i+seq_len].cpu().numpy())    # 输出目标
    X = torch.tensor(np.array(X, dtype=np.float32), device=device)  # X : (num_samples, seq_len, 3)
    Y = torch.tensor(np.array(Y, dtype=np.float32), device=device)  # Y : (num_samples, 3)
    dataset = TensorDataset(X, Y)                # 封装
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    # batch_size: 每批训练的样本数 shuffle: 是否打乱样本顺序 drop_last: 是否丢弃最后一个样本数不足的小批次

    # 初始化 GRU 模型
    model = GRUChaos(input_size=3, hidden_size=128, num_layers=3, output_size=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    start_epoch = 0
    # 尝试加载已有的模型
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device) # 读取恢复模型权重和优化器状态
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', optimizer.state_dict()))
        start_epoch = checkpoint.get('epoch', 0)
        print(f"加载已有模型 {model_path}, start_epoch={start_epoch}")
    # 尝试训练新的模型
    elif train_if_no_model:
        model.train()
        epochs = 50 # 训练50epoch
        for epoch in range(epochs):
            for xb, yb in tqdm(loader,desc=f"训练Epoch{epochs+1}/{epochs}"):
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()         # 清零梯度
                out = model(xb)               # 模型预测
                loss = nn.MSELoss()(out, yb)  # 计算均方误差MSE
                loss.backward()               # 反向传播
                optimizer.step()              # 优化器更新权重
            if (epoch+1) % 10 == 0:
                print(f"训练 epoch {epoch+1}/{epochs}, loss={loss.item():.6f}")
        torch.save({                      # 保存训练好的模型
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': float(loss.item())
        }, save_model_path)
        print(f"已保存训练好的模型到 {save_model_path}")
    else:
        print("未加载模型且训练被禁用，将使用未训练的模型权重")

    # ---------------------------
    # GRU 生成混沌序列（优化）
    # 思路：先用 seed 输入一次得到初始 hidden h，然后单步调用 GRU（seq_len=1），避免每步传入整个序列
    # ---------------------------
    model.eval()
    steps_to_generate = num_pixels
    generated = torch.zeros((steps_to_generate, 3), dtype=torch.float32, device=device)

    seed_input = X[0:1].to(device)  # shape: (1, seq_len, 3)
    with torch.no_grad():
        # 先用 seed 序列获得 hidden
        out_seed, h = model.gru(seed_input)
        last_hidden_out = out_seed[:, -1, :].to(device)  # shape: (1, hidden)
        for i in tqdm(range(steps_to_generate), desc="GRU序列生成部分"):
            pred = model.fc(last_hidden_out)  # (1, 3)
            generated[i] = pred.view(3)
            # 用 pred 作为下一步输入，传入 GRU（seq_len=1）并更新 hidden
            inp = pred.unsqueeze(1)  # (1,1,3)
            out_step, h = model.gru(inp, h)
            last_hidden_out = out_step[:, -1, :]   # 生成(num_pixels, 3)的混沌值

    # 映射为三通道密钥字节
    kr, kg, kb = seq_to_key_bytes_tensor(generated)

    # Arnold 置乱每个通道
    r_chan = torch.tensor(img_rgb[:, :, 0], device=device)
    g_chan = torch.tensor(img_rgb[:, :, 1], device=device)
    b_chan = torch.tensor(img_rgb[:, :, 2], device=device)
    r_scr = arnold_transform_channel_tensor(r_chan, iterations=arnold_iters, device=device)
    g_scr = arnold_transform_channel_tensor(g_chan, iterations=arnold_iters, device=device)
    b_scr = arnold_transform_channel_tensor(b_chan, iterations=arnold_iters, device=device)
    img_scrambled = torch.stack([r_scr, g_scr, b_scr], dim=2).to(torch.uint8)

    # 混沌扩散加密
    encrypted_rgb = chaotic_diffusion_rgb_tensor(img_scrambled, kr, kg, kb, device=device)

    metadata = {
        'arnold_iters': arnold_iters,       # Arnold迭代次数
        'num_pixels': num_pixels,           # 像素总数
        'image_shape': encrypted_rgb.shape, # 加密后的图像形状
        'model_used': model_path if model_path else save_model_path  # GRU模型路径
    }
    return encrypted_rgb.cpu().numpy(), metadata, model


# ---------------------------
# 解密流程
# ---------------------------
def decrypt_image(encrypted_rgb, input_path, password, model, metadata):
    set_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    init_state, arnold_iters = key_generation(input_path, password)
    h, w, _ = encrypted_rgb.shape
    num_pixels = h * w   # 像素总数
    rk_dt = 0.01         # 积分步长
    seq_len = 50         # GRU输入序列长度
    rk_steps = max(20000, num_pixels + seq_len + 100) # 总步数

    lorenz_seq = runge_kutta4_tensor(lorenz_system_tensor, init_state, rk_dt, rk_steps, device=device)
    X = []
    for i in tqdm(range(len(lorenz_seq) - seq_len), desc="解密GRU数据集构建部分"):
        X.append(lorenz_seq[i:i+seq_len].cpu().numpy())
    X_tensor = torch.tensor(np.array(X, dtype=np.float32), device=device)
    # 构建GRU解密密钥

    model = model.to(device)
    model.eval()    # 推理模式
    generated = torch.zeros((num_pixels, 3), dtype=torch.float32, device=device)
    input_seq = X_tensor[0:1].to(device)
    with torch.no_grad():
        out_seed, h = model.gru(input_seq)
        last_hidden_out = out_seed[:, -1, :]
        for i in tqdm(range(num_pixels), desc="GRU解密序列生成部分"):
            pred = model.fc(last_hidden_out)
            generated[i] = pred.view(3)
            inp = pred.unsqueeze(1)
            out_step, h = model.gru(inp, h)
            last_hidden_out = out_step[:, -1, :]

    kr, kg, kb = seq_to_key_bytes_tensor(generated)   # 得到密钥序列

    # 混沌逆扩散
    decrypted_scrambled = inverse_chaotic_diffusion_rgb_tensor(torch.tensor(encrypted_rgb, device=device), kr, kg, kb, device=device)
    r_scr = decrypted_scrambled[:, :, 0]
    g_scr = decrypted_scrambled[:, :, 1]
    b_scr = decrypted_scrambled[:, :, 2]

    # 逆Arnold置乱
    r_plain = inverse_arnold_transform_channel_tensor(r_scr, iterations=arnold_iters, device=device)
    g_plain = inverse_arnold_transform_channel_tensor(g_scr, iterations=arnold_iters, device=device)
    b_plain = inverse_arnold_transform_channel_tensor(b_scr, iterations=arnold_iters, device=device)

    # 合并通道输出恢复图像
    recovered_rgb = torch.stack([r_plain, g_plain, b_plain], dim=2).to(torch.uint8)
    return recovered_rgb.cpu().numpy()

# ---------------------------
# 检查图像是否完全一致
# ---------------------------
def image_equal(img1, img2):
    if img1.shape != img2.shape:
        return False
    return np.array_equal(img1, img2)

# ---------------------------
# 主程序示例
# ---------------------------
if __name__ == "__main__":
    start_time = time.time()

    set_seed(1234)
    input_path = r"D:\cxy\deeplearning\2025-2026_1\P1.jpg"
    password = "ddj"
    model_file = "gru_chaos_checkpoint.pth"
    save_dir = r"D:\cxy\deeplearning\2025-2026_1\picture"
    os.makedirs(save_dir, exist_ok=True) # 如果输出目录不存在则生成

    # 加密
    encrypted_rgb, meta, model = encrypt_image(input_path, password, model_path=model_file, train_if_no_model=False)
    print("加密完成，metadata:", meta)
    enc_bgr = cv2.cvtColor(encrypted_rgb, cv2.COLOR_RGB2BGR)
    enc_path = os.path.join(save_dir, "encrypted.jpg")
    result = cv2.imwrite(enc_path, enc_bgr)
    print("已保存加密后的图片", result)

    # 解密
    rec_rgb = decrypt_image(encrypted_rgb, input_path, password, model, meta)
    rec_bgr = cv2.cvtColor(rec_rgb, cv2.COLOR_RGB2BGR)
    rec_path = os.path.join(save_dir, "decrypted.jpg")
    result = cv2.imwrite(rec_path, rec_bgr)
    print("已保存解密后的图片", result)
    print("还原是否与裁剪后的原图完全一致?", image_equal(rec_rgb, prepare_image_for_arnold(cv2.imread(input_path))))

    end_time = time.time()
    print(f"用时{(end_time - start_time) / 60}分")