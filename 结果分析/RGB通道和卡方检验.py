import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data


def color_histogram_analysis(image_input):
    """
    对彩色图像进行直方图分析：
    - 绘制 RGB 三通道直方图
    - 计算卡方值、信息熵、均值和方差
    参数:
        image_input: str 或 np.ndarray
    """
    # 判断输入类型
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError("图像路径无效或无法读取: " + image_input)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image_input.astype(np.uint8)

    channels = ('R', 'G', 'B')
    colors = ('red', 'green', 'blue')
    results = {}

    plt.figure(figsize=(10, 8))
    plt.suptitle("RGB Histogram Analysis", fontsize=16)

    for i, (ch_name, color) in enumerate(zip(channels, colors)):
        ch_data = img_rgb[:, :, i]

        hist = cv2.calcHist([ch_data], [0], None, [256], [0, 256]).flatten()
        p = hist / np.sum(hist)
        p_nonzero = p[p > 0]
        entropy = -np.sum(p_nonzero * np.log2(p_nonzero))
        total_pixels = ch_data.size
        expected = total_pixels / 256
        chi_square = np.sum((hist - expected) ** 2 / expected)
        mean_val = np.mean(ch_data)
        std_val = np.std(ch_data)

        results[ch_name] = {
            "卡方值Chi-square": float(chi_square),
            "信息熵Entropy": float(entropy),
            "均值Mean": float(mean_val),
            "方差StdDev": float(std_val)
        }

        plt.subplot(3, 1, i + 1)
        plt.bar(range(256), hist, color=color, alpha=0.7)
        plt.xlim([0, 255])
        plt.title(f"{ch_name}-channel | χ²={chi_square:.2f}, H={entropy:.4f}, μ={mean_val:.2f}, σ={std_val:.2f}")
        plt.xlabel("Gray Level")
        plt.ylabel("Pixel Count")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return results


# 测试
from skimage import data
img_color = data.astronaut()
a = color_histogram_analysis(r"D:\cxy\deeplearning\decrypted.jpg")
print(a)
b = color_histogram_analysis(r"D:\cxy\deeplearning\encrypted.jpg")
print(b)
c = color_histogram_analysis(img_color)
print(c)