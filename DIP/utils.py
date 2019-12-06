import numpy as np

def RGB2Gray(f, mold='NTSC'):
    """ 彩色图像转灰度图像.
    
    参数:
        f: 输入图像
        mold: 变换方式
            NTSC:  coe_rgb = [0.2989, 0.5870, 0.1140]
            average: coe_rgb = [0.3333, 0.3333, 0.3333]
    
    返回值: 灰度图像, 数据类型为uint8.
    
    """
    assert(mold in ['NTSC', 'average'])
    gray = np.empty(f.shape[0:2])
    
    if mold == 'NTSC':
        coe_NTSC = np.array([0.2989, 0.5870, 0.1140])
        for row in range(f.shape[0]):
            for col in range(f.shape[1]):
                gray[row, col] = np.dot(f[row, col], coe_NTSC)
    
    elif mold == 'average':
        coe_average = np.array([0.3333, 0.3333, 0.3333])
        for row in range(f.shape[0]):
            for col in range(f.shape[1]):
                gray[row, col] = np.dot(f[row, col], coe_average)

    return gray.astype(np.uint8)

def Padding(f, hs, ws, mold='zero'):
    """　图像边界填充.

    参数: 
        f: 输入图像
        hs, ws: 填充大小
        mold: 填充类型
            zero: 0填充
            replicate: 最近邻填充
    
    返回值: 边界填充后的图像, 数据类型与f相同.

    """
    assert(mold in ['zero', 'replicate'])

    h, w = f.shape[:2]
    # 0填充
    pad = np.zeros([h+2*hs, w+2*ws], dtype=f.dtype)
    pad[hs:(hs+h), ws:(ws+w)] = f
    
    # 最近邻填充
    if mold == 'replicate':
        # 填充四个边界块
        for i in range(hs):
            pad[i,      ws:(ws+w)] = f[0, :]
            pad[hs+h+i, ws:(ws+w)] = f[h-1, :]
        for i in range(ws):
            pad[hs:(hs+h),      i] = f[:, 0]
            pad[hs:(hs+h), ws+w+i] = f[:, h-1]
        # 填充四个角
        pad[0:hs, 0:ws] = np.full([hs, ws], f[0, 0])
        pad[0:hs, (ws+w):(2*ws+w)] = np.full([hs, ws], f[0, w-1])
        pad[(hs+h):(2*hs+h), 0:ws] = np.full([hs, ws], f[h-1, 0])
        pad[(hs+h):(2*hs+h), (ws+w):(2*ws+w)] = np.full([hs, ws], f[h-1, w-1])

    return pad

def Conv2D(f, w, pad='zero'):
    """ 二维卷积函数, 使用滑窗法实现.
    
    参数:
        f: 输入图像, 要求灰度图像, 即单通道.
        w: 卷积核
        pad: 边界填充方式
            zero: 零填充
            replicate: 最近邻填充

    返回值: 卷积后的图像, 数据类型为float32.
    
    """
    assert(pad in ['zero', 'replicate'])

    # 旋转卷积核, 填充图像.
    w_rot = np.rot90(w, 2)
    f_pad = Padding(f, int((w_rot.shape[0]-1)/2), int((w_rot.shape[1]-1)/2), pad)

    h, w = f.shape
    f_conv2d = np.empty((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            f_conv2d[i, j] = np.sum(f_pad[i:i+w_rot.shape[0], j:j+w_rot.shape[1]]*w_rot)

    return f_conv2d

def Gaussian_Kernel(sig, m=-1):
    """ 生成指定参数的高斯核.
    
    参数:
        sig : 高斯核标准差
        m: 高斯核尺寸(m, m)
            -1: 当m为默认值, 高斯核尺寸由方差计算而来.
    
    返回值: 高斯核, 数据类型为float32.
    
    """
    
    if m == -1:
        m = int(np.ceil(sig * 3) * 2 + 1)
    elif m < np.ceil(sig * 3) * 2 + 1:
        print('卷积核尺寸%d过小，建议尺寸：m >= round_up(sig*3)*2+1' % m)
        exit(1)

    kernel = np.empty((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(m):
            ss2 = 2 * (sig**2)
            mid = np.ceil(sig * 3)
            kernel[i, j] = np.exp(-((i-mid)**2 + (j-mid)**2) / ss2)

    # 归一化
    kernel_sum = np.sum(kernel)
    return kernel / kernel_sum

def Threshold(f, thre):
    """ 简单的二值化处理，前景(大于阈值)为1，背景(小于或等于阈值)为0.

    参数:
        f: 输入图像, 要求是灰度图像.
        thre: 阈值

    返回值: 二值化结果, 数据类型与f相同, 但其值只有0/1.

    """
    f_flatten = f.flatten()
    binary = np.ones_like(f_flatten)
    index = np.where(f_flatten>thre)[0]
    binary[index] = 0
    
    return binary.reshape(f.shape)

def DFT2D(f):
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

def IDFT2D(F):
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

def FFTShift(F):
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
