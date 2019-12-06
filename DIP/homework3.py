import numpy as np
import matplotlib.pyplot as plt
import sys

def dft2D(f):
    """ 二维离散傅里叶变换
    """

    row, col = f.shape
    
    Fxv = np.empty([row, col], dtype=np.complex)
    Fuv = np.empty([row, col], dtype=np.complex)
    for i in range(row):
        Fxv[i, :] = np.fft.fft(f[i, :])
    for i in range(col):
        Fuv[:, i] = np.fft.fft(Fxv[:, i])
    
    return Fuv

def idft2D(F):
    """ 二维离散傅里叶逆变换
    """

    row, col = F.shape

    F = np.conjugate(F) # 先求F的共轭
    fuy = np.empty([row, col], dtype=np.complex)
    fxy = np.empty([row, col], dtype=np.complex)
    for i in range(row):
        fuy[i, :] = np.fft.fft(F[i, :])
    for i in range(col):
        fxy[:, i] = np.fft.fft(fuy[:, i])
    
    return np.conjugate(fxy/(row*col))

def fftshift(F):
    """ 中心化
    示例 : 
        F = [[1 2 3]
             [4 5 6]
             [7 8 9]]
        return = [[9 7 8]
                  [3 1 2]
                  [6 4 5]]
    """

    M, N = F.shape
    Mmid = int(np.around(M/2-0.4))
    Nmid = int(np.around(N/2-0.4))
    Mmid_ = int(np.around(M/2+0.4))
    Nmid_ = int(np.around(N/2+0.4))

    Fshift = np.empty([M, N], dtype=F.dtype)
    Fshift[0:Mmid, 0:Nmid] = F[Mmid_:M, Nmid_:N]
    Fshift[0:Mmid, Nmid:N] = F[Mmid_:M, 0:Nmid_]
    Fshift[Mmid:M, 0:Nmid] = F[0:Mmid_, Nmid_:N]
    Fshift[Mmid:M, Nmid:N] = F[0:Mmid_, 0:Nmid_]

    return Fshift

if __name__ == '__main__':
    """ 读取图片并归一化灰度值范围
    """
    default_path = './rose512.tif'
    argv = list(sys.argv)
    if len(argv) == 1:
        path = default_path
    else:
        path = argv[1]

    print('Read Image :', path)
    img = plt.imread(path)
    img = img / 255

    plt.subplot(2, 4, 1)
    plt.title('f')
    plt.imshow(img, 'gray')

    """ 问题3
    """
    # 对rose.tif做二维离散傅里叶变换，并将变换结果中心化、取对数，然后以图像方式显示。
    F = dft2D(img)
    FShift = fftshift(F)
    plt.subplot(2, 4, 2)
    plt.title('F')
    plt.imshow(np.log(1+np.abs(FShift)), 'gray')

    # 对rose.tif的def2D结果做逆变换，复原图像。
    g = idft2D(F)
    plt.subplot(2, 4, 3)
    plt.title('g')
    plt.imshow(np.abs(g.real), 'gray')

    d = np.abs(img - np.abs(g.real))
    #print(d[200:216, 200:216]) # 直接查看差值图像的部分值。
    plt.subplot(2, 4, 4)
    plt.title('d')
    print('max in d\n', d.max())
    plt.imshow(d, 'gray', vmin=0, vmax=1)

    """ 问题4
    """
    # 生成原图像。
    rect = np.zeros([512, 512])
    rect[226:286, 251:261] = 1
    plt.subplot(2, 4, 5)
    plt.title('rect')
    plt.imshow(rect, 'gray')

    # 计算rect的二维傅里叶变换。
    Frect = dft2D(rect)
    plt.subplot(2, 4, 6)
    plt.title('Frect')
    plt.imshow(np.abs(Frect), 'gray')

    # 获得并绘制中心化的谱图像。
    FrectShift = fftshift(Frect)
    plt.subplot(2, 4, 7)
    plt.title('FrectShift')
    plt.imshow(np.abs(FrectShift), 'gray')

    # 对中心化的谱图像做对数变换并绘制结果。
    plt.subplot(2, 4, 8)
    plt.title('FShiftLog')
    plt.imshow(np.log(1+np.abs(FrectShift)), 'gray')

    plt.show()

    """
    a = np.array([0, 0, 0, 3, 3, 3, 0, 0, 0]).reshape(3, 3)
    print('dft2D\n', dft2D(a))
    print('np.fft2\n', np.fft.fft2(a))
    print('idft2D\n', idft2D(dft2D(a)))
    print('np.ifft2\n', np.fft.ifft2(np.fft.fft2(a)))
    """