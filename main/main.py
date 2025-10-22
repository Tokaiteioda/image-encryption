# gru_image_encrypt_gpu.py
import os
import hashlib
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# ---------------------------
# 固定随机种子，保证可复现
# ---------------------------
def set_seed(seed=12345):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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

    def normalized(b):
        val_int = int.from_bytes(b, 'big')
        val_max = 2 ** (8 * len(b)) - 1
        return val_int / val_max

    x0 = normalized(x_bytes)
    y0 = normalized(y_bytes)
    z0 = normalized(z_bytes)

    # 用 hash 的前两个字节生成 Arnold 迭代次数（1~30）
    iter_seed = xor_hash[0] + xor_hash[1] * 256
    arnold_iters = (iter_seed % 30) + 1

    return [x0, y0, z0], arnold_iters

# ---------------------------
# Lorenz 混沌系统（GPU 计算）
# ---------------------------
def lorenz_system_tensor(state, a=10, b=8/3, c=28):
    """输入 state: (3,) 张量, 输出 dx, dy, dz"""
    x, y, z = state
    dx = a * (y - x)
    dy = x * (c - z) - y
    dz = x * y - b * z
    return torch.stack([dx, dy, dz])

# ---------------------------
# GPU 版 RK4 积分生成混沌轨迹
# ---------------------------
def runge_kutta4_tensor(system, initial_state, h, steps, device='cuda'):
    """Lorenz 系统积分, 返回 shape = (steps,3) 的张量轨迹"""
    state = torch.tensor(initial_state, dtype=torch.float32, device=device)
    traj = torch.zeros((steps, 3), dtype=torch.float32, device=device)

    for i in tqdm(range(steps), desc="Lorenz RK4部分"):
        traj[i] = state
        k1 = h * system(state)
        k2 = h * system(state + 0.5 * k1)
        k3 = h * system(state + 0.5 * k2)
        k4 = h * system(state + k3)
        state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

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
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  # 输出映射到 [0,1]
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        # 仅取最后一个时间步作为输出
        out = self.fc(out[:, -1, :])
        return out

# ---------------------------
# 图像裁剪到中心正方形
# ---------------------------
def crop_center_square(img):
    h, w = img.shape[:2]
    if h == w:
        return img
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    return img[top:top+side, left:left+side]

# ---------------------------
# GPU 版 Arnold 变换（单通道）
# ---------------------------
def arnold_transform_channel_tensor(channel, iterations=1, a=1, b=1):
    """输入 channel: (N,N) 张量, 返回置乱后的张量"""
    N = channel.shape[0]
    res = channel.clone()
    for _ in range(iterations):
        new = torch.zeros_like(res)
        for x in range(N):
            for y in range(N):
                x_new = (x + a * y) % N
                y_new = (b * x + (a * b + 1) * y) % N
                new[x_new, y_new] = res[x, y]
        res = new
    return res

def inverse_arnold_transform_channel_tensor(channel, iterations=1, a=1, b=1):
    """逆 Arnold 变换"""
    N = channel.shape[0]
    res = channel.clone()
    for _ in range(iterations):
        new = torch.zeros_like(res)
        for x in range(N):
            for y in range(N):
                # M = [[1,1],[1,2]] 对应 a=b=1
                x_new = (2 * x - y) % N
                y_new = (-x + y) % N
                new[x_new, y_new] = res[x, y]
        res = new
    return res

# ---------------------------
# 将 GRU 输出浮点序列映射到 uint8 密钥字节
# ---------------------------
def seq_to_key_bytes_tensor(generated_seq, num_bytes=1):
    """输入 generated_seq: (N,3) 张量, 返回 kr, kg, kb"""
    seq = torch.clamp(generated_seq, 0.0, 1.0)
    combo_r = (seq[:, 0] + seq[:, 1]) % 1.0
    combo_g = (seq[:, 1] + seq[:, 2]) % 1.0
    combo_b = (seq[:, 2] + seq[:, 0]) % 1.0

    def to_bytes(arr):
        val = torch.floor((arr * 1e6) % 256).to(torch.uint8)
        return val

    kr = to_bytes(combo_r)
    kg = to_bytes(combo_g)
    kb = to_bytes(combo_b)
    return kr, kg, kb

# ---------------------------
# Chaotic Diffusion（加密/解密）
# ---------------------------
def chaotic_diffusion_rgb_tensor(img_rgb, kr, kg, kb):
    """输入 img_rgb: (H,W,3) uint8 张量, kr,kg,kb: 1D 张量"""
    h, w, _ = img_rgb.shape
    num = h * w
    flat = img_rgb.reshape(-1,3).to(torch.int64)
    cipher = torch.zeros_like(flat, dtype=torch.uint8, device=flat.device)
    c_prev = torch.zeros(3, dtype=torch.int64, device=flat.device)
    for i in tqdm(range(num),desc="混沌扩散部分"):
        p = flat[i]
        k = torch.tensor([int(kr[i]), int(kg[i]), int(kb[i])], dtype=torch.int64, device=flat.device)
        c = (p + k + c_prev) % 256
        cipher[i] = c.to(torch.uint8)
        c_prev = c
    return cipher.reshape(h,w,3)

def inverse_chaotic_diffusion_rgb_tensor(cipher_rgb, kr, kg, kb):
    h, w, _ = cipher_rgb.shape
    num = h * w
    flat = cipher_rgb.reshape(-1,3).to(torch.int64)
    plain = torch.zeros_like(flat, dtype=torch.uint8, device=flat.device)
    c_prev = torch.zeros(3, dtype=torch.int64, device=flat.device)
    for i in tqdm(range(num),desc="混沌逆扩散部分"):
        c = flat[i]
        k = torch.tensor([int(kr[i]), int(kg[i]), int(kb[i])], dtype=torch.int64, device=flat.device)
        p = (c - k - c_prev) % 256
        plain[i] = p.to(torch.uint8)
        c_prev = c
    return plain.reshape(h,w,3)

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
    set_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取图像
    img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图片: {input_path}")
    img_rgb = prepare_image_for_arnold(img_bgr)
    h, w = img_rgb.shape[:2]
    num_pixels = h * w

    # 生成初始密钥和 Arnold 迭代次数
    init_state, arnold_iters = key_generation(input_path, password)

    # Lorenz + RK4 生成训练序列
    lorenz_steps = max(rk_steps, num_pixels + seq_len + 100)
    lorenz_seq = runge_kutta4_tensor(lorenz_system_tensor, init_state, rk_dt, lorenz_steps, device=device)

    # 构建 GRU 数据集
    X, Y = [], []
    for i in range(len(lorenz_seq) - seq_len):
        X.append(lorenz_seq[i:i+seq_len].cpu().numpy())
        Y.append(lorenz_seq[i+seq_len].cpu().numpy())
    X = torch.tensor(np.array(X, dtype=np.float32), device=device)
    Y = torch.tensor(np.array(Y, dtype=np.float32), device=device)
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    # 初始化 GRU 模型
    model = GRUChaos(input_size=3, hidden_size=128, num_layers=3, output_size=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    start_epoch = 0
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', optimizer.state_dict()))
        start_epoch = checkpoint.get('epoch', 0)
        print(f"加载已有模型 {model_path}, start_epoch={start_epoch}")
    elif train_if_no_model:
        model.train()
        epochs = 50
        for epoch in range(epochs):
            for xb, yb in tqdm(loader,desc=f"训练Epoch{epochs+1}/{epochs}"):
                optimizer.zero_grad()
                out = model(xb)
                loss = nn.MSELoss()(out, yb)
                loss.backward()
                optimizer.step()
            if (epoch+1) % 10 == 0:
                print(f"训练 epoch {epoch+1}/{epochs}, loss={loss.item():.6f}")
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': float(loss.item())
        }, save_model_path)
        print(f"已保存训练好的模型到 {save_model_path}")
    else:
        print("未加载模型且训练被禁用，将使用未训练的模型权重（不推荐）")

    # GRU 生成混沌序列
    model.eval()
    generated = []
    seed_input = X[0:1].to(device)
    input_seq = seed_input.clone()
    steps_to_generate = num_pixels
    with torch.no_grad():
        for _ in tqdm(range(steps_to_generate),desc="GRU序列生成部分"):
            pred = model(input_seq)
            generated.append(pred.cpu().numpy().flatten())
            next_in = pred.unsqueeze(1)
            input_seq = torch.cat((input_seq[:, 1:, :], next_in), dim=1)
    generated = torch.tensor(np.array(generated, dtype=np.float32), device=device)

    # 映射为密钥字节
    kr, kg, kb = seq_to_key_bytes_tensor(generated)

    # Arnold 置乱每个通道
    r_chan = torch.tensor(img_rgb[:, :, 0], device=device)
    g_chan = torch.tensor(img_rgb[:, :, 1], device=device)
    b_chan = torch.tensor(img_rgb[:, :, 2], device=device)
    r_scr = arnold_transform_channel_tensor(r_chan, iterations=arnold_iters)
    g_scr = arnold_transform_channel_tensor(g_chan, iterations=arnold_iters)
    b_scr = arnold_transform_channel_tensor(b_chan, iterations=arnold_iters)
    img_scrambled = torch.stack([r_scr, g_scr, b_scr], dim=2).to(torch.uint8)

    # 混沌扩散加密
    encrypted_rgb = chaotic_diffusion_rgb_tensor(img_scrambled, kr, kg, kb)

    metadata = {
        'arnold_iters': arnold_iters,
        'num_pixels': num_pixels,
        'image_shape': encrypted_rgb.shape,
        'model_used': model_path if model_path else save_model_path
    }
    return encrypted_rgb.cpu().numpy(), metadata, model

# ---------------------------
# 解密流程
# ---------------------------
def decrypt_image(encrypted_rgb, input_path, password, model, metadata):
    set_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    init_state, arnold_iters = key_generation(input_path, password)
    h, w, _ = encrypted_rgb.shape
    num_pixels = h * w
    rk_dt = 0.01
    seq_len = 50
    rk_steps = max(20000, num_pixels + seq_len + 100)

    lorenz_seq = runge_kutta4_tensor(lorenz_system_tensor, init_state, rk_dt, rk_steps, device=device)
    X = []
    for i in range(len(lorenz_seq) - seq_len):
        X.append(lorenz_seq[i:i+seq_len].cpu().numpy())
    X_tensor = torch.tensor(np.array(X, dtype=np.float32), device=device)

    model.eval()
    generated = []
    input_seq = X_tensor[0:1].to(device)
    with torch.no_grad():
        for _ in range(num_pixels):
            pred = model(input_seq)
            generated.append(pred.cpu().numpy().flatten())
            input_seq = torch.cat((input_seq[:, 1:, :], pred.unsqueeze(1)), dim=1)
    generated = torch.tensor(np.array(generated, dtype=np.float32), device=device)
    kr, kg, kb = seq_to_key_bytes_tensor(generated)

    decrypted_scrambled = inverse_chaotic_diffusion_rgb_tensor(torch.tensor(encrypted_rgb, device=device), kr, kg, kb)
    r_scr = decrypted_scrambled[:, :, 0]
    g_scr = decrypted_scrambled[:, :, 1]
    b_scr = decrypted_scrambled[:, :, 2]

    r_plain = inverse_arnold_transform_channel_tensor(r_scr, iterations=arnold_iters)
    g_plain = inverse_arnold_transform_channel_tensor(g_scr, iterations=arnold_iters)
    b_plain = inverse_arnold_transform_channel_tensor(b_scr, iterations=arnold_iters)

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
    set_seed(1234)
    input_path = r"F:\PycharmProjects\pythonProject\tuxiangjiami\main\Original image\P1.jpg"
    password = "hajimi"
    model_file = "gru_chaos_checkpoint.pth"
    save_dir = r"F:\PycharmProjects\pythonProject\tuxiangjiami\main\Generated images"
    os.makedirs(save_dir, exist_ok=True)

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
