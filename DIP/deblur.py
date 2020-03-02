import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from utils import Padding, Subplot
from wtfilter import Wtfilter


def Kernel(shape, distance, angle):
    """ 由模糊核的尺度和角度信息生成目标尺寸的模糊核.

    参数:
        shape: 模糊核尺寸
        distance: 模糊核尺度
        angle: 模糊核角度

    """
    kernel = np.zeros(shape)
    center = np.array([(shape[0]-1)/2, (shape[1]-1)/2]) # 模糊核图像中心点.

    coef = np.array([-np.sin(angle*np.pi/180), np.cos(angle*np.pi/180)])
    for i in range(distance):
        # 对运动方向上的像素赋值, 全1.
        index = center+coef*i
        kernel[int(index[0]), int(index[1])] = 1

    # 归一化.
    return kernel / kernel.sum()

def Wiener(f, kernel, K=0.01):
    """ 维纳滤波
    """
    F = np.fft.fft2(f)
    kernel_fft2 = np.fft.fft2(kernel)
    result = np.fft.ifft2(F * np.conj(kernel_fft2) / (np.abs(kernel_fft2) ** 2 + K))
    return np.fft.fftshift(result.real)

def Local_min(f, u, v):
    """ 判断f[u, v]是否是局部极小值.
    """
    if f[u, v] < f[u-1, v] and f[u, v] < f[u, v-1] and\
       f[u, v] < f[u, v+1] and f[u, v] < f[u+1, v]:
        return True
    else:
        return False

def Search(cepstrum):
    """ 使用查找特殊点的方法估计模糊核的尺度和角度.
    """
    f = cepstrum.copy()
    idx = np.argmax(cepstrum)
    centeru = idx // 512
    centerv = idx % 512

    r = 1
    minu = -1
    minv = -1
    while True:
        for i in range(r+1):
            if Local_min(f, centeru-r, centerv-i):
                minu = centeru-r
                minv = centerv-i
            if Local_min(f, centeru-r, centerv+i):
                minu = centeru-r
                minv = centerv+i
            if Local_min(f, centeru+r, centerv-i):
                minu = centeru+r
                minv = centerv-i
            if Local_min(f, centeru+r, centerv+i):
                minu = centeru+r
                minv = centerv+i

            if Local_min(f, centeru-i, centerv-r):
                minu = centeru-i
                minv = centerv-r
            if Local_min(f, centeru+i, centerv-r):
                minu = centeru+i
                minv = centerv-r
            if Local_min(f, centeru-i, centerv+r):
                minu = centeru-i
                minv = centerv+r
            if Local_min(f, centeru+i, centerv+r):
                minu = centeru+i
                minv = centerv+r

        if minu != -1:
                break
        r += 1

    distance = np.sqrt((centeru-minu)**2 + (centerv-minv)**2)
    angle = np.arctan(abs(centeru-minu)/abs(centerv-minv))*180/np.pi

    f[centeru, centerv] = 0
    f[minu, minv] = np.max(f)
    f[centeru-minu+centeru, centerv-minv+centerv] = np.max(f)

    return distance, angle, f

if __name__ == '__main__':
    fig = plt.figure()
    
    img = io.imread("image/lena.png")
    Subplot(fig, img, 2, 4, 1, 'original image')

    # 计算并绘制倒谱.
    img_fft2 = np.fft.fft2(img)
    cepstrum = np.fft.ifft2(1+abs(img_fft2))
    cepstrum_shift = np.fft.fftshift(abs(cepstrum))
    Subplot(fig, cepstrum_shift, 2, 4, 2, 'cepstrum')

    distance, angle, new_cepstrum_shift = Search(cepstrum_shift)
    Subplot(fig, new_cepstrum_shift[256-26:256+26, 256-26:256+26], 2, 4, 3, 'search result')

    # 估计并绘制模糊核.
    kernel = Kernel(img.shape, int(distance), 180-round(angle))
    Subplot(fig, kernel[256-26:256+26, 256-26:256+26], 2, 4, 4, 'kernel')
    
    # 去模糊, 并交互地选择维纳滤波中的常数项k.
    deblur001 = Wiener(img, kernel, 0.01)
    deblur0001 = Wiener(img, kernel, 0.001)
    deblur000001 = Wiener(img, kernel, 0.00001)
    lena_resconstruct_5 = Wtfilter(deblur000001, 7, 'bior4.4') # 小波域维纳滤波处理k取较小值得到的结果.

    Subplot(fig, deblur001, 2, 4, 5, 'deblur k=1e-2')
    Subplot(fig, deblur0001, 2, 4, 6, 'deblur k=1e-3')
    Subplot(fig, deblur000001, 2, 4, 7, 'deblur k=1e-5')
    Subplot(fig, lena_resconstruct_5, 2, 4, 8, 'bior4.4')

    plt.show()

    fig2 = plt.figure()
    Subplot(fig2, deblur001[280:360, 230:310], 1, 4, 1, 'deblur k=1e-2')
    Subplot(fig2, deblur0001[280:360, 230:310], 1, 4, 2, 'deblur k=1e-3')
    Subplot(fig2, deblur000001[280:360, 230:310], 1, 4, 3, 'deblur k=1e-5')
    Subplot(fig2, lena_resconstruct_5[280:360, 230:310], 1, 4, 4, 'bior4.4')

    plt.show()
