import utils
import matplotlib.pyplot as plt
import numpy as np
import sys

img_path = list(sys.argv)[1]
print(list(sys.argv))
rgbimg = plt.imread(img_path)
if len(rgbimg.shape) == 3: 
    grayimg = utils.rgb2gray(rgbimg)
else:
    grayimg = rgbimg
plt.subplot(2, 4, 1)
plt.title('gray')
plt.imshow(grayimg, 'gray')
plt.xticks([]), plt.yticks([])

w_1 = utils.gaussKernel(1)
w_2 = utils.gaussKernel(2)
w_3 = utils.gaussKernel(3)
w_5 = utils.gaussKernel(5)

r_1 = utils.twodConv(grayimg, w_1)
plt.subplot(2, 4, 2)
plt.title('sig 1')
plt.imshow(r_1, 'gray')
plt.xticks([]), plt.yticks([])

r_2 = utils.twodConv(grayimg, w_2)
plt.subplot(2, 4, 3)
plt.title('sig 2')
plt.imshow(r_2, 'gray')
plt.xticks([]), plt.yticks([])

r_3 = utils.twodConv(grayimg, w_3)
plt.subplot(2, 4, 4)
plt.title('sig 3')
plt.imshow(r_3, 'gray')
plt.xticks([]), plt.yticks([])

r_5 = utils.twodConv(grayimg, w_5)
plt.subplot(2, 4, 5)
plt.title('sig 5')
plt.imshow(r_5, 'gray')
plt.xticks([]), plt.yticks([])

from cv2 import cv2
r_cv = cv2.GaussianBlur(grayimg, (7, 7), 1)
plt.subplot(2, 4, 6)
plt.title('sig 1 opencv')
plt.imshow(r_cv, 'gray')
plt.xticks([]), plt.yticks([])

dif = r_cv - r_1
plt.subplot(2, 4, 7)
plt.title('opencv - sig 1')
plt.imshow(dif, 'gray')
plt.xticks([]), plt.yticks([])
#print(dif[100:116, 100:116])

r_1_replicate = utils.twodConv(grayimg, w_1, 'replicate')
dif_2 = r_1_replicate - r_1
plt.subplot(2, 4, 8)
plt.title('replicate - zero')
plt.imshow(dif_2, 'gray')
plt.xticks([]), plt.yticks([])
#print(dif_2[100:116, 100:116])

plt.show()