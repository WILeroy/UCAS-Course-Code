import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import sys

def calcCDF(hist, mn):
    """ 计算直方图均衡化的映射数组
    参数:
        hist : 直方图
        level : 灰度级
    返回值:
        cdf : 映射数组
    """

    cumsum = np.cumsum(hist)
    const_a = 255 / mn
    cdf = const_a * cumsum
    return np.around(cdf)

def histequal4e(I):
    """ 直方图均衡化
    参数:
        I : 输入图像
    返回值:
        equalI : 均衡化之后的图像
    """

    shape = I.shape
    hist = np.zeros([256], dtype=np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            hist[I[i, j]] += 1

    cdf = calcCDF(hist, mn=shape[0]*shape[1])

    equalI = np.zeros(shape, dtype=np.uint8)
    equalI = cdf[I]

    return equalI
 
def noise(I, p=0.05):
    """ 生成椒盐噪声
    """ 
    noise_img = np.copy(I)
    shape = noise_img.shape
    num = int(shape[0] * shape[1] * p)
    for i in range(num):
        w = np.random.randint(0, shape[1]-1)
        h = np.random.randint(0, shape[0]-1)
        if np.random.randint(0, 2) == 0:
            noise_img[h, w] = 0
        else:
            noise_img[h, w] = 255
    return noise_img

def padding(f, hs, ws, mold='zero'):
    """　边界填充
    参数 : 
        f : 输入图像
        hs, ws : 填充大小
        mold : 填充类型
            zero : 零填充
            replicate : 最近邻填充
    """
    assert(mold in ['zero', 'replicate'])

    pad = np.zeros([f.shape[0]+2*hs, f.shape[1]+2*ws])
    pad[hs:(hs+f.shape[0]), ws:(ws+f.shape[1])] = f
    
    if mold == 'replicate':

        for i in range(hs):
            pad[i,               ws:(ws+f.shape[1])] = f[0, :]
            pad[hs+f.shape[0]+i, ws:(ws+f.shape[1])] = f[f.shape[0]-1, :]
        
        for i in range(ws):
            pad[hs:(hs+f.shape[0]),               i] = f[:, 0]
            pad[hs:(hs+f.shape[0]), ws+f.shape[1]+i] = f[:, f.shape[0]-1]

        pad[0:hs, 0:ws] = np.full([hs, ws], f[0, 0])
        pad[0:hs, (ws+f.shape[1]):(2*ws+f.shape[1])] = np.full([hs, ws], f[0, f.shape[1]-1])
        pad[(hs+f.shape[0]):(2*hs+f.shape[0]), 0:ws] = np.full([hs, ws], f[f.shape[0]-1, 0])
        pad[(hs+f.shape[0]):(2*hs+f.shape[0]), (ws+f.shape[1]):(2*ws+f.shape[1])] =\
            np.full([hs, ws], f[f.shape[0]-1, f.shape[1]-1])

    return pad

def selective_edge_smooth(I):
    """ 有选择保边缘平滑法
    """
    kernel_3_3 = np.array([1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9], dtype=np.float32).reshape([3, 3])
    
    kernel_5_1 = np.array([1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 0, 1/7, 0], dtype=np.float32).reshape([3, 3])
    kernel_5_2 = np.array([0, 1/7, 1/7, 1/7, 1/7, 1/7, 0, 1/7, 1/7], dtype=np.float32).reshape([3, 3])
    kernel_5_3 = np.array([0, 1/7, 0, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7], dtype=np.float32).reshape([3, 3])
    kernel_5_4 = np.array([1/7, 1/7, 0, 1/7, 1/7, 1/7, 1/7, 1/7, 0], dtype=np.float32).reshape([3, 3])
    
    kernel_6_1 = np.array([0, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 0], dtype=np.float32).reshape([3, 3])
    kernel_6_2 = np.array([1/7, 1/7, 0, 1/7, 1/7, 1/7, 0, 1/7, 1/7], dtype=np.float32).reshape([3, 3])
    kernel_6_3 = np.array([0, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 0], dtype=np.float32).reshape([3, 3])
    kernel_6_4 = np.array([1/7, 1/7, 0, 1/7, 1/7, 1/7, 0, 1/7, 1/7], dtype=np.float32).reshape([3, 3])
    
    h, w = I.shape[:2]
    padI = padding(I, 2, 2).astype(np.float32)
    
    # 正方形
    mean_3_3 = np.empty([h, w, 1], dtype=np.float32)
    var_3_3 = np.empty([h, w, 1], dtype=np.float32)
    for i in range(h):
        for j in range(w):
            mean_3_3[i, j] = np.sum(padI[i+1:i+4, j+1:j+4] * kernel_3_3)
            var_3_3[i, j] = np.sum((padI[i+1:i+4, j+1:j+4]-mean_3_3[i, j]) ** 2 * kernel_3_3)
    
    # 五边形
    mean_5_1 = np.empty([h, w, 1], dtype=np.float32)
    var_5_1 = np.empty([h, w, 1], dtype=np.float32)
    for i in range(h):
        for j in range(w):
            mean_5_1[i, j] = np.sum(padI[i:i+3, j+1:j+4] * kernel_5_1)
            var_5_1[i, j] = np.sum((padI[i:i+3, j+1:j+4]-mean_5_1[i, j]) ** 2 * kernel_5_1)
    var = np.concatenate([var_3_3, var_5_1], axis=2)
    mean = np.concatenate([mean_3_3, mean_5_1], axis=2)

    mean_5_2 = np.empty([h, w, 1], dtype=np.float32)
    var_5_2 = np.empty([h, w, 1], dtype=np.float32)
    for i in range(h):
        for j in range(w):
            mean_5_2[i, j] = np.sum(padI[i+1:i+4, j+2:j+5] * kernel_5_2)
            var_5_2[i, j] = np.sum((padI[i+1:i+4, j+2:j+5]-mean_5_2[i, j]) ** 2 * kernel_5_2)
    var = np.concatenate([var, var_5_2], axis=2)
    mean = np.concatenate([mean, mean_5_2], axis=2)

    mean_5_3 = np.empty([h, w, 1], dtype=np.float32)
    var_5_3 = np.empty([h, w, 1], dtype=np.float32)
    for i in range(h):
        for j in range(w):
            mean_5_3[i, j] = np.sum(padI[i+2:i+5, j+1:j+4] * kernel_5_3)
            var_5_3[i, j] = np.sum((padI[i+2:i+5, j+1:j+4]-mean_5_3[i, j]) ** 2 * kernel_5_3)
    var = np.concatenate([var, var_5_3], axis=2)
    mean = np.concatenate([mean, mean_5_3], axis=2)

    mean_5_4 = np.empty([h, w, 1], dtype=np.float32)
    var_5_4 = np.empty([h, w, 1], dtype=np.float32)
    for i in range(h):
        for j in range(w):
            mean_5_4[i, j] = np.sum(padI[i+1:i+4, j:j+3] * kernel_5_4)
            var_5_4[i, j] = np.sum((padI[i+1:i+4, j:j+3]-mean_5_4[i, j]) ** 2 * kernel_5_4)
    var = np.concatenate([var, var_5_4], axis=2)
    mean = np.concatenate([mean, mean_5_4], axis=2)

    # 六边形
    mean_6_1 = np.empty([h, w, 1], dtype=np.float32)
    var_6_1 = np.empty([h, w, 1], dtype=np.float32)
    for i in range(h):
        for j in range(w):
            mean_6_1[i, j] = np.sum(padI[i:i+3, j+2:j+5] * kernel_6_1)
            var_6_1[i, j] = np.sum((padI[i:i+3, j+2:j+5]-mean_6_1[i, j]) ** 2 * kernel_6_1)
    var = np.concatenate([var, var_6_1], axis=2)
    mean = np.concatenate([mean, mean_6_1], axis=2)

    mean_6_2 = np.empty([h, w, 1], dtype=np.float32)
    var_6_2 = np.empty([h, w, 1], dtype=np.float32)
    for i in range(h):
        for j in range(w):
            mean_6_2[i, j] = np.sum(padI[i+2:i+5, j+2:j+5] * kernel_6_2)
            var_6_2[i, j] = np.sum((padI[i+2:i+5, j+2:j+5]-mean_6_2[i, j]) ** 2 * kernel_6_2)
    var = np.concatenate([var, var_6_2], axis=2)
    mean = np.concatenate([mean, mean_6_2], axis=2)

    mean_6_3 = np.empty([h, w, 1], dtype=np.float32)
    var_6_3 = np.empty([h, w, 1], dtype=np.float32)
    for i in range(h):
        for j in range(w):
            mean_6_3[i, j] = np.sum(padI[i+2:i+5, j:j+3] * kernel_6_3)
            var_6_3[i, j] = np.sum((padI[i+2:i+5, j:j+3]-mean_6_3[i, j]) ** 2 * kernel_6_3)
    var = np.concatenate([var, var_6_3], axis=2)
    mean = np.concatenate([mean, mean_6_3], axis=2)

    mean_6_4 = np.empty([h, w, 1], dtype=np.float32)
    var_6_4 = np.empty([h, w, 1], dtype=np.float32)
    for i in range(h):
        for j in range(w):
            mean_6_4[i, j] = np.sum(padI[i:i+3, j:j+3] * kernel_6_4)
            var_6_4[i, j] = np.sum((padI[i:i+3, j:j+3]-mean_6_4[i, j]) ** 2 * kernel_6_4)
    var = np.concatenate([var, var_6_4], axis=2)
    mean = np.concatenate([mean, mean_6_4], axis=2)

    # 计算结果
    result = np.empty([h, w])
    min_var_index = np.argmin(var, axis=2)
    for i in range(h):
        for j in range(w):
            result[i, j] = mean[i, j, min_var_index[i, j]]
    return result, var, mean

def laplacian(I, pad_mold='zero'):
    """ 拉普拉斯增强
    参数 : 
        I : 输入图像
        mold : 填充类型
            zero : 零填充
            replicate : 最近邻填充
    返回值 :
        laplacianI : 增强之后的图像
    """
    kernel = np.array([0, -1, 0, -1, 5, -1, 0, -1, 0]).reshape([3, 3])

    h, w = I.shape[:2]
    padI = padding(I, 1, 1, mold=pad_mold)
    laplacianI = np.empty([h, w])
    for i in range(h):
        for j in range(w):
            laplacianI[i, j] = np.sum(padI[i:i+3, j:j+3] * kernel)

    return np.clip(laplacianI, 0, 255).astype(np.uint8)

if __name__ == '__main__':
    # 设置默认命令行参数
    default_path = 'cameraman.tif'
    argv = list(sys.argv)
    if len(argv) == 1:
        path = default_path
    else:
        path = argv[1]
    img = io.imread(default_path)

    # -------------------------- 第一题 ---------------------
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(img, 'gray')
    plt.xticks([]), plt.yticks([])

    img_equal = histequal4e(img)
    plt.subplot(1, 2, 2)
    plt.title('Equal hist image')
    plt.imshow(img_equal, 'gray')
    plt.xticks([]), plt.yticks([])
    plt.show()

    plt.subplot(2, 1, 1)
    plt.hist(img.flatten(), bins=256)
    plt.title('Original hist')

    plt.subplot(2, 1, 2)
    plt.hist(img_equal.flatten(), bins=256)
    plt.title('Equal Hist')
    plt.show()
    
    # -------------------------- 第二题 ---------------------
    plt.subplot(1, 3, 1)
    plt.title('Original image')
    plt.imshow(img, 'gray')
    plt.xticks([]), plt.yticks([])

    noise_img = noise(img)
    plt.subplot(1, 3, 2)
    plt.title('Noise image')
    plt.imshow(noise_img, 'gray')
    plt.xticks([]), plt.yticks([])
    
    import time
    start = time.time()
    smooth, var, mean = selective_edge_smooth(noise_img)
    end = time.time()
    plt.subplot(1, 3, 3)
    plt.title('Smooth image {:.2f}s'.format(end-start))
    plt.imshow(smooth, 'gray')
    plt.xticks([]), plt.yticks([])
    plt.show()
    

    # -------------------------- 第三题 ---------------------
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(img, 'gray')
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, 2, 2)
    laplacianI = laplacian(img)
    plt.title('Laplacian image')
    plt.imshow(laplacianI, 'gray')
    plt.xticks([]), plt.yticks([])
    plt.show()