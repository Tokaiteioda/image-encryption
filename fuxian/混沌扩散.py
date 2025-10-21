import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 设置 Matplotlib 使用 TkAgg 后端
import matplotlib.pyplot as plt

def gru_chaotic_diffusion(image, chaotic_seq):
    """
    使用 GRU 生成的混沌序列进行扩散
    :param image: 原图（二维 np.uint8）
    :param chaotic_seq: GRU 输出的混沌序列 (N,3)
    """
    h, w = image.shape
    img = image.flatten()
    seq = ((chaotic_seq[:, 0] + chaotic_seq[:, 1] + chaotic_seq[:, 2]) * 1e6 % 256).astype(np.uint8)
    cipher = np.zeros_like(img, dtype=np.uint8)

    cipher[0] = (img[0] + seq[0]) % 256
    for i in range(1, len(img)):
        cipher[i] = (img[i] + seq[i] + cipher[i-1]) % 256

    return cipher.reshape(h, w)


def inverse_gru_diffusion(cipher, chaotic_seq):
    """逆扩散，用于解密"""
    h, w = cipher.shape
    img = cipher.flatten()
    seq = ((chaotic_seq[:, 0] + chaotic_seq[:, 1] + chaotic_seq[:, 2]) * 1e6 % 256).astype(np.uint8)
    plain = np.zeros_like(img, dtype=np.uint8)

    plain[0] = (img[0] - seq[0]) % 256
    for i in range(1, len(img)):
        plain[i] = (img[i] - seq[i] - img[i-1]) % 256

    return plain.reshape(h, w)


if __name__ == "__main__":
    import cv2

    img = cv2.imread("picture.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为 RGB

