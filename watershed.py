# coding=utf-8
# 分水岭算法分割图像
import numpy as np
import cv2
from matplotlib import pyplot as plt
# 加载图像
img = cv2.imread('images/basil.jpg')
# 转化为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 设置阈值（二值化），将图像分为两部分：黑色部分和白色部分
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# 移除噪声
kernel = np.ones((3, 3), np.uint8) # 创建 3 * 3 卷积核
# 进行形态变换
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
# 膨胀操作，确定背景区域
sure_bg = cv2.dilate(opening, kernel, iterations = 3)

# 获取前景区域
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
# 应用阈值来决定前景
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# 获取重合（交叉）区域
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# 标记连接的背景区域
ret, markers = cv2.connectedComponents(sure_fg)

# 通过将标记加1，将背景区域标记为1
markers = markers + 1

# 将重合（未知）区域标记为 0
markers[unknown == 255] = 0
# 放水，使用分水岭算法计算出“栅栏”，并描绘为红色
markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]
# 显示结果
plt.imshow(img)
plt.show()
