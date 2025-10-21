import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 设置 Matplotlib 使用 TkAgg 后端
import matplotlib.pyplot as plt

def preprocessing(picture_path):  # 图像预检验
    img = cv2.imread(picture_path)
    height, width = img.shape[:2]  # 高度,长度
    if height != width:
        print("长宽不相等")

def arnold_transform(image, iterations=5):
    N = image.shape[0]
    result = np.copy(image)

    for _ in range(iterations):
        new_img = np.zeros_like(result)
        for x in range(N):
            for y in range(N):
                x_new = (x + y) % N
                y_new = (x + 2 * y) % N
                new_img[x_new, y_new] = result[x, y]
        result = new_img
    return result

def inverse_arnold_transform(image, iterations=5):
    N = image.shape[0]
    result = np.copy(image)

    for _ in range(iterations):
        new_img = np.zeros_like(result)
        for x in range(N):
            for y in range(N):
                x_new = (2 * x - y) % N
                y_new = (-x + y) % N
                new_img[x_new, y_new] = result[x, y]
        result = new_img
    return result

if __name__ == "__main__":
    from matplotlib import rcParams

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    img = cv2.imread("picture.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为 RGB
    scrambled = arnold_transform(img)
    restored = inverse_arnold_transform(scrambled)

    plt.figure(figsize=(12,4))
    plt.subplot(1, 3, 1)
    plt.title("原图")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("变换后")
    plt.imshow(scrambled)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("复原后")
    plt.imshow(restored)
    plt.axis("off")

    plt.tight_layout()
    plt.show()