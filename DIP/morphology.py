"""
该部分是图像处理中的一些形态学算法, 包括:

- 膨胀
- 腐蚀
- 细化
- 裁剪
- 距离变换

Author: WILeroy
Date: 2019.12.7
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from utils import Padding, Threshold, RGB2Gray, Subplot


def Expansion(f, kernel):
    """ 膨胀算法
    参数:
        f: 输入图像, numpy数组
        kernel: 膨胀核
    
    返回值:
        膨胀结果, 数据类型与输入图像一致.
    """
    padf = Padding(f, 1, 1)
    newf = np.zeros_like(f)
    shape = f.shape
    for r in range(shape[0]):
        for c in range(shape[1]):
            if np.sum(padf[r:r+3, c:c+3] * kernel):
                newf[r, c] = 1

    return newf

def Erode(f, kernel):
    """ 腐蚀算法，腐蚀核提供不考虑功能
    kernel:
        -1: 不考虑
        0: 背景
        1: 前景
    """
    padf = Padding(f, 1, 1)
    kernel_noi = np.ones_like(kernel.flatten())
    kernel_noi[np.where(kernel.flatten()==-1)[0]] = 0
    kernel_noi = kernel_noi.reshape([3, 3])

    newf = np.empty_like(f)
    shape = f.shape
    for r in range(shape[0]):
        for c in range(shape[1]):
            if np.sum(abs((padf[r:r+3, c:c+3] - kernel) * kernel_noi)):
                newf[r, c] = 0
            else:
                newf[r, c] = 1

    return newf

def Thinning(f):
    """ 细化算法求骨架
    """
    kernel = []
    kernel.append(np.array([0, 0, 0, -1, 1, -1, 1, 1, 1]).reshape([3, 3]))
    kernel.append(np.array([-1, 0, 0, 1, 1, 0, 1, 1, -1]).reshape([3, 3]))
    kernel.append(np.array([1, -1, 0, 1, 1, 0, 1, -1, 0]).reshape([3, 3]))
    kernel.append(np.array([1, 1, -1, 1, 1, 0, -1, 0, 0]).reshape([3, 3]))

    kernel.append(np.array([1, 1, 1, -1, 1, -1, 0, 0, 0]).reshape([3, 3]))
    kernel.append(np.array([-1, 1, 1, 0, 1, 1, 0, 0, -1]).reshape([3, 3]))
    kernel.append(np.array([0, -1, 1, 0, 1, 1, 0, -1, 1]).reshape([3, 3]))
    kernel.append(np.array([0, 0, -1, 0, 1, 1, -1, 1, 1]).reshape([3, 3]))

    copyf = np.copy(f)
    lastf = np.copy(f)
    
    while 1:
        for i in range(8):
            copyf = copyf - Erode(copyf, kernel[i])
        if np.sum(copyf-lastf):
            lastf = np.copy(copyf)
        else:
            break

    return copyf

def DisTransform(f):
    """ 距离变换
    f: 输入的二值图像，前景为1，背景为0
    """
    shape = f.shape
    padf = np.ones([f.shape[0]+2, f.shape[1]+2], dtype=np.int32) * 65535
    padf[1:f.shape[0]+1, 1:f.shape[1]+1] = ((f+1) % 2 * 65534 + 1)

    for r in range(1, shape[0]+1):
        for c in range(1, shape[1]+1):
            dlist = [padf[r-1, c-1]+4, padf[r-1, c]+3, padf[r-1, c+1]+4, padf[r, c-1]+3, padf[r, c]]
            #dlist = [padf[r-1, c-1]+1, padf[r-1, c]+1, padf[r-1, c+1]+1, padf[r, c-1]+1, padf[r, c]]
            padf[r, c] = min(dlist)

    for r in range(shape[0], 0, -1):
        for c in range(shape[1], 0, -1):
            dlist = [padf[r+1, c+1]+4, padf[r+1, c]+3, padf[r+1, c-1]+4, padf[r, c+1]+3, padf[r, c]]
            #dlist = [padf[r+1, c+1]+1, padf[r+1, c]+1, padf[r+1, c-1]+1, padf[r, c+1]+1, padf[r, c]]
            padf[r, c] = min(dlist)
    
    return padf[1:shape[0]+1, 1:shape[1]+1]

def Get_border(f):
    kernel = np.ones([9]).reshape([3, 3])
    border = f - Erode(f, kernel)
    return border

def Local_max(f):
    """ 求f的局部最大值，将取得局部最大值的元素赋值为1
    """
    shape = f.shape
    padf = Padding(f, 1, 1)
    kernel = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape([3, 3]) #选择局部区域
    result = np.zeros(shape, dtype=np.uint8)
    for r in range(shape[0]):
        for c in range(shape[1]):
            if f[r, c] == np.max(padf[r:r+3, c:c+3]*kernel):
                result[r, c] = 1
    return result

def Cut(f, erode_num, expansion_num):
    """ 裁剪算法
    erode_num: 腐蚀（细化）次数
    expansion_num: 膨胀次数
    """
    kernel = []
    kernel.append(np.array([-1, 0, 0, 1, 1, 0, -1, 0, 0]).reshape([3, 3]))
    kernel.append(np.array([-1, 1, -1, 0, 1, 0, 0, 0, 0]).reshape([3, 3]))
    kernel.append(np.array([0, 0, -1, 0, 1, 1, 0, 0, -1]).reshape([3, 3]))
    kernel.append(np.array([0, 0, 0, 0, 1, 0, -1, 1, -1]).reshape([3, 3]))
    kernel.append(np.array([1, 0, 0, 0, 1, 0, 0, 0, 0]).reshape([3, 3]))
    kernel.append(np.array([0, 0, 1, 0, 1, 0, 0, 0, 0]).reshape([3, 3]))
    kernel.append(np.array([0, 0, 0, 0, 1, 0, 0, 0, 1]).reshape([3, 3]))
    kernel.append(np.array([0, 0, 0, 0, 1, 0, 1, 0, 0]).reshape([3, 3]))

    x1 = np.copy(f)
    for count in range(erode_num):
        for i in range(8):
            x1 = x1 - Erode(x1, kernel[i])

    x2 = np.zeros_like(f)
    for i in range(8):
        x2 = x2 + Erode(x1, kernel[i])

    x3 = np.copy(x2)
    for i in range(expansion_num):
        x3 = Expansion(x3, np.ones([3, 3])) * f

    # 细化结果与膨胀结果取并集
    shape = f.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if x3[i, j]:
                x1[i, j] = 1

    return x1

if __name__ == '__main__':

    image = io.imread('./image/smallfingerprint.jpg')
    gray_img = RGB2Gray(image)


    """ 形态学算法求骨架
    """
    binary = Threshold(gray_img, 160)    # 1-二值化
    thin = Thinning(binary)              # 2-细化求骨架
    cut_thin = Cut(thin, 4, 2)           # 3-裁剪

    fig = plt.figure()
    Subplot(fig, binary, 1, 3, 1, 'binary')
    Subplot(fig, thin, 1, 3, 2, 'thin')
    Subplot(fig, cut_thin, 1, 3, 3, 'cut')
    plt.show()


    """ 距离变换求骨架
    """
    border = Get_border(binary)    # 4-1-计算边界
    disT = DisTransform(border)    # 4-2-距离变换（可视化时乘上二值化掩模）
    lmax = Local_max(disT)         # 4-3-局部最大值提取骨架
    lmax = lmax * binary

    fig = plt.figure()
    Subplot(fig, binary, 1, 4, 1, 'binary')
    Subplot(fig, border, 1, 4, 2, 'border')
    Subplot(fig, disT, 1, 4, 3, 'disTransform')
    Subplot(fig, lmax, 1, 4, 4, 'local max')
    plt.show()


    """ 对比不同方法所得骨架图像的相同局部
    """
    fig = plt.figure()
    Subplot(fig, binary[162:, 100:], 1, 4, 1, 'binary')
    Subplot(fig, thin[162:, 100:], 1, 4, 2, 'thin')
    Subplot(fig, cut_thin[162:, 100:], 1, 4, 3, 'cut')
    Subplot(fig, lmax[162:, 100:], 1, 4, 4, 'local max')
    plt.show()
