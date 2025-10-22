import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 设置 Matplotlib 使用 TkAgg 后端
import matplotlib.pyplot as plt


def correlation_analysis_rgb(image_path, num_points=5000):
    """
    对彩色图像 (RGB) 进行相邻像素相关性分析（水平、垂直、对角）
    - 每个通道单独分析
    - 绘制散点图可视化
    - 输出每个通道的相关系数结果

    参数:
        image_path: 图像路径
        num_points: 随机采样的像素点数

    返回:
        corr_results: dict 形式 {通道: {方向: 相关系数}}
    """
    # 读取图像 (BGR -> RGB)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"无法读取图像：{image_path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # 随机采样坐标
    x = np.random.randint(0, h - 1, num_points)
    y = np.random.randint(0, w - 1, num_points)

    # 三个方向
    directions = {
        '水平相邻Horizontal': (x, y, x, y + 1),
        '垂直相邻Vertical': (x, y, x + 1, y),
        '对角相邻Diagonal': (x, y, x + 1, y + 1)
    }

    # 通道名称
    channels = ['R', 'G', 'B']
    corr_results = {}

    # 可视化绘图
    plt.figure(figsize=(12, 10))
    plot_index = 1

    for c_idx, c_name in enumerate(channels):
        corr_results[c_name] = {}
        channel_data = img[:, :, c_idx]

        for direction, (x1, y1, x2, y2) in directions.items():
            # 防止越界
            valid_mask = (x2 < h) & (y2 < w)
            a = channel_data[x1[valid_mask], y1[valid_mask]]
            b = channel_data[x2[valid_mask], y2[valid_mask]]

            # 计算相关系数
            r = np.corrcoef(a, b)[0, 1]
            corr_results[c_name][direction] = r

            # 绘制散点图
            plt.subplot(3, 3, plot_index)
            plt.scatter(a, b, s=1, alpha=0.5, c=c_name.lower())
            plt.title(f"{c_name}-{direction}\nCorr = {r:.4f}")
            plt.xlabel("Pixel(i)")
            plt.ylabel("Pixel(j)")
            plt.grid(True)
            plot_index += 1

    plt.suptitle("RGB 相邻像素相关性分析", fontsize=16)
    plt.tight_layout()
    plt.show()

    # 打印结果
    print("\n=== 各通道相关系数结果 ===")
    for c, vals in corr_results.items():
        print(f"\n通道 {c}:")
        for d, r in vals.items():
            print(f"  {d:<10}: {r:.4f}")

    return corr_results


# 示例使用
if __name__ == "__main__":
    path = r"F:\PycharmProjects\pythonProject\tuxiangjiami\main\Generated images\encrypted.jpg"  # 图像路径
    correlation_analysis_rgb(path)