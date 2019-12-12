import matplotlib.pyplot as plt
import numpy as np
import pywt
from skimage import io

from utils import RGB2Gray, Subplot, Padding

def Wtfilter(f, level, wavelet):
    """ 小波域维纳滤波.
    
    参数:
        f: 滤波图像
        level: 小波分解层数
        wavelet: 小波种类

    """
    coef_list = []
    f_reconstruct = f.copy()
    
    # 分解
    for i in range(level):
        LL, (LH, HL, HH) = pywt.dwt2(f_reconstruct, wavelet)
        coef_list.append((LL, (LH, HL, HH)))
        f_reconstruct = LL
        if i == 0:
            HH_array = HH.reshape(HH.shape[0]*HH.shape[1])
        else:
            HH_array = np.concatenate((HH_array, HH.reshape(HH.shape[0]*HH.shape[1])))

    # 计算噪声方差
    sigma_n_2 = (np.median(abs(HH_array)) / 0.6745) ** 2
    
    # 滤波与重构
    for i in range(level):
        LL, (LH, HL, HH) = coef_list[level-1-i]
        
        sigma_lh_2 = np.sum(LH ** 2) / (LH.shape[0] * LH.shape[1]) - sigma_n_2
        sigma_hl_2 = np.sum(HL ** 2) / (HL.shape[0] * HL.shape[1]) - sigma_n_2
        sigma_hh_2 = np.sum(HH ** 2) / (HH.shape[0] * HH.shape[1]) - sigma_n_2

        LH *= (sigma_lh_2 / (sigma_lh_2 + sigma_n_2))
        HL *= (sigma_hl_2 / (sigma_hl_2 + sigma_n_2))
        HH *= (sigma_hh_2 / (sigma_hh_2 + sigma_n_2))

        f_reconstruct = f_reconstruct[:HH.shape[0], :HH.shape[1]]
        f_reconstruct = pywt.idwt2((f_reconstruct, (LH, HL, HH)), wavelet)
    return f_reconstruct

if __name__ == '__main__':
    # 读取图像, 并转化为灰度图像
    lena = io.imread('./DIP/image/lena512color.tiff')
    lena_gray = RGB2Gray(lena).astype(np.float64)
    fig = plt.figure()
    Subplot(fig, lena_gray, 2, 3, 1, 'lena')

    # 显示噪声图像
    gauss_noise = np.random.normal(0, 0.05 ** 0.5, lena_gray.shape)
    lena_noise = lena_gray.copy() / 255
    lena_noise += gauss_noise
    lena_noise *= 255
    Subplot(fig, lena_noise, 2, 3, 2, 'lena+noise')

    # 全局方差小波域维纳滤波
    lena_resconstruct_1 = Wtfilter(lena_noise, 1, 'bior4.4')
    lena_resconstruct_2 = Wtfilter(lena_noise, 2, 'bior4.4')
    lena_resconstruct_3 = Wtfilter(lena_noise, 3, 'bior4.4')
    lena_resconstruct_4 = Wtfilter(lena_noise, 4, 'bior4.4')
    lena_resconstruct_5 = Wtfilter(lena_noise, 5, 'bior4.4')
    lena_resconstruct_6 = Wtfilter(lena_noise, 6, 'bior4.4')

    Subplot(fig, lena_resconstruct_1, 2, 3, 4, '1')
    Subplot(fig, lena_resconstruct_3, 2, 3, 5, '3')
    Subplot(fig, lena_resconstruct_5, 2, 3, 6, '5')

    # 局部方差小波域维纳滤波
    
    M = 512 ** 2
    error = []
    error.append(np.sum(abs(lena_gray - lena_noise)) / M)
    error.append(np.sum(abs(lena_gray - lena_resconstruct_1)) / M)
    error.append(np.sum(abs(lena_gray - lena_resconstruct_2)) / M)
    error.append(np.sum(abs(lena_gray - lena_resconstruct_3)) / M)
    error.append(np.sum(abs(lena_gray - lena_resconstruct_4)) / M)
    error.append(np.sum(abs(lena_gray - lena_resconstruct_5)) / M)
    error.append(np.sum(abs(lena_gray - lena_resconstruct_6)) / M)
    ax = fig.add_subplot(2, 3, 3)
    ax.plot([0, 1, 2, 3, 4, 5, 6], error)
    ax.plot([0, 1, 2, 3, 4, 5, 6], error, 'ro')
    ax.set_title('Evaluate')
    ax.set_ylabel('Error')
    ax.set_xlabel('Number of decomposition')

    fig.tight_layout()
    plt.show()
