# coding=utf-8
import numpy as np
import cv2
from matplotlib import pyplot as plt

# 加载用来查询和检测的图片
img1 = cv2.imread('images/manowar_logo.png', 0)
img2 = cv2.imread('images/manowar_single.jpg', 0)

# 创建ORB检测对象
orb = cv2.ORB_create()
# 检测关键点并计算描述符
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 创建BF匹配器
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
# 根据描述符进行KNN匹配,无需排序
matches = bf.knnMatch(des1, des2, k = 10)

# 绘制KNN匹配结果
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, img2, flags = 2)
# 显示匹配结果
plt.imshow(img3), plt.show()