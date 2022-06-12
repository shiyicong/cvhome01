import numpy as np
import random
import cv2
from matplotlib import pyplot as plt


def sp_nosie(img, prob):
    """添加椒盐噪声
    prob信噪比"""
    output = np.zeros(img.shape, np.uint8)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]

    return output


def gasuss_nosie(img, mean=0, var=0.01):
    """添加高斯噪声 mean表示均值 var表示方差"""
    img = np.array(img / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    out = img + noise
    if out.min() < 0:
        low_clip = -1
    else:
        low_clip = 0
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out


def meanFiltering1(img, size):  # img输入，size均值滤波器的尺寸，>=3，且必须为奇数
    """"高斯滤波"""
    num = int((size - 1) / 2)  # 输入图像需要填充的尺寸
    img = cv2.copyMakeBorder(img, num, num, num, num, cv2.BORDER_REPLICATE)  # 对传入的图像进行扩充，尺寸为num
    h1, w1 = img.shape[0:2]
    # 高斯滤波
    img1 = np.zeros((h1, w1, 3), dtype="uint8")  # 定义空白图像，用于输出中值滤波后的结果
    for i in range(num, h1 - num):  # 对扩充图像中的原图进行遍历
        for j in range(num, w1 - num):
            sum = 0
            sum1 = 0
            sum2 = 0
            for k in range(i - num, i + num + 1):  # 求中心像素周围size*size区域内的像素的平均值
                for l in range(j - num, j + num + 1):
                    sum = sum + img[k, l][0]  # B通道
                    sum1 = sum1 + img[k, l][1]  # G通道
                    sum2 = sum2 + img[k, l][2]  # R通道
            sum = sum / (size ** 2)  # 除以核尺寸的平方
            sum1 = sum1 / (size ** 2)
            sum2 = sum2 / (size ** 2)
            img1[i, j] = [sum, sum1, sum2]  # 复制给空白图像
    img1 = img1[(0 + num):(h1 - num), (0 + num):(h1 - num)]  # 从滤波图像中裁剪出原图像
    return img1


img = cv2.imread('E:\PythonProject\cvhome01\data\R-C.jpg')
"""原图"""
# cv2.imshow('origin', img)
"""添加椒盐噪声,信噪比为0.01"""
img_spNoise = sp_nosie(img, 0.01)
"""添加高斯噪声"""
img_gasussNoise = gasuss_nosie(img)


# """中值滤波"""
# """对添加椒盐噪声的中值滤波"""
# img_medsp = cv2.medianBlur(img_spNoise, 5)
# cv2.imshow('sp_med', img_medsp)
# cv2.imwrite('sp_med.jpg', img_medsp)
# """对添加高斯噪声的中值滤波"""
# img_medgas = cv2.medianBlur(img, 5)
# cv2.imshow('gas_med', img_medgas)
# cv2.imwrite('gas_med.jpg', img_medgas)

# """高斯滤波"""
# """对添加椒盐噪声的高斯滤波"""
# img_gasOfsp = cv2.GaussianBlur(img_spNoise, (3, 3), 0)
# cv2.imshow('gas_sp', img_gasOfsp)
# cv2.imwrite('gas_sp.jpg', img_gasOfsp)
# """对添加高斯噪声的高斯滤波"""
# img_gasOfgas = cv2.GaussianBlur(img_gasussNoise, (3, 3), 0)
# cv2.imshow('gas_gas', img_gasOfgas)
# cv2.imwrite('gas_gas.jpg', img_gasOfgas)

# """保留细节的滤波——双边滤波"""
# """椒盐噪声"""
# img_spOfbil = cv2.bilateralFilter(img_spNoise, 0, 10, 10)
# cv2.imshow('spOfbil', img_spOfbil)
"""均值滤波"""
img_spOfmean = cv2.blur(img_spNoise, (3, 3))
cv2.imwrite('sp_mean.jpg',img_spOfmean)
img_gasOfmean = cv2.blur(img_gasussNoise,(3,3))
cv2.imwrite('gas_mean.jpg', img_gasOfmean)
cv2.waitKey(0)