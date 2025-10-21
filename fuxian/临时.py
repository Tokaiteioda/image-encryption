# gru_image_encrypt.py
import os
import hashlib
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ---------------------------
# 种子保证可复现
# ---------------------------
def set_seed(seed=12345):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------
# 生成密钥 (SHA-256)
# ---------------------------
def key_generation(picture_path, password):
    """从图片和口令生成 3 值初始状态（0~1）和一个迭代次数种子"""
    with open(picture_path, 'rb') as f:
        data = f.read()
    hash_value = hashlib.sha256(data).digest()
    password_value = hashlib.sha256(password.encode()).digest()
    xor_hash = bytes(h ^ p for h, p in zip(hash_value, password_value))

    # 切分为三段用于初始 x0,y0,z0
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

    # 用 hash 的前两个字节决定 Arnold 迭代次数（可作为密钥的一部分）
    iter_seed = xor_hash[0] + xor_hash[1] * 256
    arnold_iters = (iter_seed % 30) + 1  # 1 ~ 30 次，可自调

    return [x0, y0, z0], arnold_iters

# ---------------------------
# Lorenz 混沌系统 + RK4
# ---------------------------
def lorenz_system(state, a=10, b=8/3, c=28):
    x, y, z = state
    dx = a * (y - x)
    dy = x * (c - z) - y
    dz = x * y - b * z
    return np.array([dx, dy, dz], dtype=float)

def runge_kutta4(system, initial_state, h, steps):
    state = np.array(initial_state, dtype=float)
    traj = np.zeros((steps, 3), dtype=float)
    for i in range(steps):
        traj[i] = state
        k1 = h * system(state)
        k2 = h * system(state + 0.5 * k1)
        k3 = h * system(state + 0.5 * k2)
        k4 = h * system(state + k3)
        state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    # 每通道归一化到 [0,1]
    def normalize_col(col):
        mn = np.min(col)
        mx = np.max(col)
        if mx - mn == 0:
            return np.zeros_like(col)
        return (col - mn) / (mx - mn)

    x_norm = normalize_col(traj[:, 0])
    y_norm = normalize_col(traj[:, 1])
    z_norm = normalize_col(traj[:, 2])
    return np.stack([x_norm, y_norm, z_norm], axis=1)  # (steps, 3)

# ---------------------------
# GRU 模型
# ---------------------------
class GRUChaos(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=3, output_size=3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# ---------------------------
# Arnold 变换
# ---------------------------
def crop_center_square(img):
    h, w = img.shape[:2]
    if h == w:
        return img
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    return img[top:top+side, left:left+side]

def arnold_transform_channel(channel, iterations=1, a=1, b=1):
    """对单通道做 Arnold 置乱（方阵）"""
    N = channel.shape[0]
    res = channel.copy()
    for _ in range(iterations):
        new = np.zeros_like(res)
        for x in range(N):
            for y in range(N):
                x_new = (x + a * y) % N
                y_new = (b * x + (a * b + 1) * y) % N
                new[x_new, y_new] = res[x, y]
        res = new
    return res

def inverse_arnold_transform_channel(channel, iterations=1, a=1, b=1):
    N = channel.shape[0]
    res = channel.copy()
    for _ in range(iterations):
        new = np.zeros_like(res)
        # 求逆矩阵在整数模 N 下的映射式（对 a=b=1，上面给出特例）
        for x in range(N):
            for y in range(N):
                # 逆映射解：
                # [x,y]^T = M^{-1} [x',y']^T mod N
                # 对 a=b=1，有 M = [[1,1],[1,2]], M^{-1} = [[2,-1],[-1,1]]
                x_new = (2 * x - y) % N
                y_new = (-x + y) % N
                new[x_new, y_new] = res[x, y]
        res = new
    return res

# ---------------------------
# 将 GRU 生成的实值序列映射到图像
# ---------------------------
def seq_to_key_bytes(generated_seq, num_bytes=1):
    """
    将 (N,3) 的浮点序列映射到 0-255 的 uint8 流。
    我们把三个通道作组合映射，得到更复杂分布。
    """
    # 保证 0~1
    seq = np.clip(generated_seq, 0.0, 1.0)
    # 组合方式可调整
    combo_r = (seq[:, 0] + seq[:, 1])  # R 组合
    combo_g = (seq[:, 1] + seq[:, 2])  # G 组合
    combo_b = (seq[:, 2] + seq[:, 0])  # B 组合

    # 扩展放大并模 256
    def to_bytes(arr):
        val = np.floor((arr * 1e6) % 256).astype(np.uint8)
        return val

    kr = to_bytes(combo_r)
    kg = to_bytes(combo_g)
    kb = to_bytes(combo_b)
    return kr, kg, kb  # 每个长度为像素数

# ---------------------------
# Chaotic diffusion (forward and inverse)
# ---------------------------
def chaotic_diffusion_rgb(img_rgb, kr, kg, kb):
    """
    img_rgb: HxWx3 uint8
    kr,kg,kb: 1D arrays len = num_pixels
    Uses C[i] = (P[i] + K[i] + C[i-1]) % 256 (row-major flatten)
    """
    h, w, _ = img_rgb.shape
    num = h * w
    flat = img_rgb.reshape((-1, 3)).astype(np.int64)  # shape (num,3)
    cipher = np.zeros_like(flat, dtype=np.uint8)

    # first pixel
    c_prev = np.array([0,0,0], dtype=np.int64)
    for i in range(num):
        p = flat[i]
        k = np.array([int(kr[i]), int(kg[i]), int(kb[i])], dtype=np.int64)
        c = (p + k + c_prev) % 256
        cipher[i] = c.astype(np.uint8)
        c_prev = c
    return cipher.reshape((h,w,3))

def inverse_chaotic_diffusion_rgb(cipher_rgb, kr, kg, kb):
    h, w, _ = cipher_rgb.shape
    num = h * w
    flat = cipher_rgb.reshape((-1, 3)).astype(np.int64)
    plain = np.zeros_like(flat, dtype=np.uint8)

    # 解密 P[i] = (C[i] - K[i] - C[i-1]) % 256
    c_prev = np.array([0,0,0], dtype=np.int64)
    for i in range(num):
        c = flat[i]
        k = np.array([int(kr[i]), int(kg[i]), int(kb[i])], dtype=np.int64)
        p = (c - k - c_prev) % 256
        plain[i] = p.astype(np.uint8)
        c_prev = c
    return plain.reshape((h,w,3))

# ---------------------------
# 加密 / 解密
# ---------------------------
def prepare_image_for_arnold(img_bgr):
    """将BGR转换为RGB，通过中心裁剪确保正方形，返回rgb uint8"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    square = crop_center_square(rgb)
    return square

def encrypt_image(input_path, password, model_path=None, save_model_path='gru_chaos_checkpoint.pth',
                  train_if_no_model=False, rk_steps=20000, rk_dt=0.01, seq_len=50):
    """
    完整加密流水线（训练模型可选）
    - input_path: 图片路径
    - password: 用户口令
    - model_path: 如果已有训练好的模型则加载，否则可选择训练临时模型（train_if_no_model=True）
    Returns: encrypted_rgb (np.uint8),  dict格式
    """
    set_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 读取并预处理图像
    img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图片: {input_path}")
    img_rgb = prepare_image_for_arnold(img_bgr)
    h, w = img_rgb.shape[:2]
    num_pixels = h * w

    # 2) 生成密钥初值和 arnold 迭代次数
    init_state, arnold_iters = key_generation(input_path, password)

    # 3) 用 Lorenz + RK4 生成训练序列（可以比 num_pixels 更长用于训练）
    lorenz_steps = max(rk_steps, num_pixels + seq_len + 100)
    lorenz_seq = runge_kutta4(lambda s: lorenz_system(s, a=10, b=8/3, c=28), init_state, rk_dt, lorenz_steps)

    # 4) 准备 GRU 数据集（滑动窗口）
    data = lorenz_seq  # shape (lorenz_steps, 3)
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len])
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    X_tensor = torch.tensor(X).to(device)
    Y_tensor = torch.tensor(Y).to(device)
    dataset = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    # 5) 准备或加载模型
    model = GRUChaos(input_size=3, hidden_size=128, num_layers=3, output_size=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    start_epoch = 0
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', optimizer.state_dict()))
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Loaded model from {model_path}, start_epoch={start_epoch}")
    elif train_if_no_model:
        # 训练一个简单模型
        model.train()
        epochs = 50
        for epoch in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = nn.MSELoss()(out, yb)
                loss.backward()
                optimizer.step()
            if (epoch+1) % 10 == 0:
                print(f"Train epoch {epoch+1}/{epochs}, loss={loss.item():.6f}")
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': float(loss.item())
        }, save_model_path)
        print(f"Saved trained model to {save_model_path}")
    else:
        print("No model loaded and training disabled; using untrained model weights (NOT recommended).")

    # 6) 生成新的混沌序列：长度 = num_pixels
    model.eval()
    generated = []
    # 从数据集中选择一个起始窗口
    seed_input = X_tensor[0:1].to(device)  # shape (1, seq_len, 3)
    input_seq = seed_input.clone()
    steps_to_generate = num_pixels
    with torch.no_grad():
        for _ in range(steps_to_generate):
            pred = model(input_seq)  # (1,3)
            generated.append(pred.cpu().numpy().flatten())
            # 窗口滑动
            next_in = pred.unsqueeze(1)  # (1,1,3)
            input_seq = torch.cat((input_seq[:, 1:, :], next_in), dim=1)
    generated = np.array(generated, dtype=np.float32)  # (num_pixels, 3)

    # 映射生成的序列 -> 每个通道的关键字节数
    kr, kg, kb = seq_to_key_bytes(generated)

    # Arnold 置乱各通道（用 arnold_iters，从key派生）
    # 将 rgb 各通道分离并置乱
    r_chan = img_rgb[:, :, 0]
    g_chan = img_rgb[:, :, 1]
    b_chan = img_rgb[:, :, 2]

    r_scr = arnold_transform_channel(r_chan, iterations=arnold_iters)
    g_scr = arnold_transform_channel(g_chan, iterations=arnold_iters)
    b_scr = arnold_transform_channel(b_chan, iterations=arnold_iters)

    img_scrambled = np.stack([r_scr, g_scr, b_scr], axis=2).astype(np.uint8)

    # 9) 混沌扩散 (使用 kr,kg,kb)
    encrypted_rgb = chaotic_diffusion_rgb(img_scrambled, kr, kg, kb)

    metadata = {
        'arnold_iters': arnold_iters,
        'num_pixels': num_pixels,
        'image_shape': encrypted_rgb.shape,
        'model_used': model_path if model_path else save_model_path
    }
    return encrypted_rgb, metadata, model  # 返回 model 以便保存/进一步使用

def decrypt_image(encrypted_rgb, input_path, password, model, metadata):
    """
    解密流程：
    - 依据 input_path + password 重新生成密钥（init state, arnold_iters）
    - 用 model 生成相同的密钥流 kr,kg,kb（保证训练/模型一致）
    - 先逆扩散，再逆 Arnold
    """
    set_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (init_state, arnold_iters) = key_generation(input_path, password)
    # 生成同样的 Lorenz 和 GRU 序列
    h, w, _ = encrypted_rgb.shape
    num_pixels = h * w
    # 使用与加密时相同的 RK 参数 / seq_len
    rk_dt = 0.01
    seq_len = 50
    rk_steps = max(20000, num_pixels + seq_len + 100)

    lorenz_seq = runge_kutta4(lambda s: lorenz_system(s, a=10, b=8/3, c=28), init_state, rk_dt, rk_steps)
    # 构建数据集和种子输入与 encrypt_image 完全相同
    X = []
    for i in range(len(lorenz_seq) - seq_len):
        X.append(lorenz_seq[i:i+seq_len])
    X = np.array(X, dtype=np.float32)
    X_tensor = torch.tensor(X).to(device)

    # 生成序列
    model.eval()
    generated = []
    input_seq = X_tensor[0:1].to(device)
    with torch.no_grad():
        for _ in range(num_pixels):
            pred = model(input_seq)
            generated.append(pred.cpu().numpy().flatten())
            input_seq = torch.cat((input_seq[:, 1:, :], pred.unsqueeze(1)), dim=1)
    generated = np.array(generated, dtype=np.float32)

    kr, kg, kb = seq_to_key_bytes(generated)

    # 逆拡散
    decrypted_scrambled = inverse_chaotic_diffusion_rgb(encrypted_rgb, kr, kg, kb)

    # 逆Arnold
    r_scr = decrypted_scrambled[:, :, 0]
    g_scr = decrypted_scrambled[:, :, 1]
    b_scr = decrypted_scrambled[:, :, 2]

    r_plain = inverse_arnold_transform_channel(r_scr, iterations=arnold_iters)
    g_plain = inverse_arnold_transform_channel(g_scr, iterations=arnold_iters)
    b_plain = inverse_arnold_transform_channel(b_scr, iterations=arnold_iters)

    recovered_rgb = np.stack([r_plain, g_plain, b_plain], axis=2).astype(np.uint8)
    return recovered_rgb

# ---------------------------
# 验证是否完全复原
# ---------------------------
def image_equal(img1, img2):
    if img1.shape != img2.shape:
        return False
    return np.array_equal(img1, img2)

# ---------------------------
# 运行
# ---------------------------
if __name__ == "__main__":
    set_seed(1234)
    input_path = r"C:\Users\ltteo\Desktop\picture.jpg"  # 图片路径
    password = "hajimi"
    model_file = "gru_chaos_checkpoint.pth"

    save_dir = r"F:\PycharmProjects\pythonProject\tuxiangjiami\fuxian\picture"
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在,如果不存在自动生成

    # 尝试加载已有模型（如果没有，可设置 train_if_no_model=True 进行训练）
    encrypted_rgb, meta, model = encrypt_image(input_path, password, model_path=model_file,
                                               train_if_no_model=False)
    print("加密完成，metadata:", meta)
    # 将结果写出（BGR 保存）
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
