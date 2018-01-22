# coding=utf-8
import numpy as np 
import cv2
from matplotlib import pyplot as plt
# 加载图像
img = cv2.imread('images/statue_small.jpg')
print img.shape
print img.shape[:2]
# 创建一个与所加载图像同形状的掩模，用0填充
mask = np.zeros(img.shape[:2], np.uint8)
# 创建以0填充的前景和背景模型
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
# 定义用于分离对象的矩形区域
rect = (100, 50, 421, 378)
# 使用指定的空模型（背景和前景模型）运行GrabCut算法进行像素分类（分离前景和背景），最大迭代次数为5
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
# 将mask中值为0和2的像素点转为0，其它（值为1和3）的转为1，并以unit8的矩阵保存到mask2中
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
# 用mask2过滤出所有0值像素，从而保留前景像素，实习前景和背景对象的分离
img = img * mask2[:, :, np.newaxis]
# 通过pyplot显示结果
plt.subplot(121), plt.imshow(img)
plt.title("Grabcut"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cv2.cvtColor(cv2.imread('images/statue_small.jpg'), cv2.COLOR_BGR2RGB))
plt.title("Original"), plt.xticks([]), plt.yticks([])
plt.show()