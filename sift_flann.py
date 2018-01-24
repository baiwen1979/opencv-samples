# coding=utf-8
import numpy as np
import cv2
from matplotlib import pyplot as plt

# 加载要查询的图像
queryImage = cv2.imread('images/bathory_album.jpg', 0)
# 加载训练（被查询）的图像
trainingImage = cv2.imread('images/bathory_vinyls.jpg', 0)

# 创建SIFT对象并进行检测和计算
sift = cv2.xfeatures2d.SIFT_create()
# 检测和计算要查询图像的关键点和描述符
kp1, des1 = sift.detectAndCompute(queryImage, None)
# 检测和计算训练（被查询）图像中的关键点和描述符
kp2, des2 = sift.detectAndCompute(trainingImage, None)

# 定义FLANN匹配器参数（字典）
# FLANN_INDEX_KDTREE = 0
#  定义索引树参数
indexParams = dict(algorithm = 0, trees = 5)
#  定义索引树要被遍历的次数
searchParams = dict(checks = 50)

# 创建FLANN匹配器对象
flann = cv2.FlannBasedMatcher(indexParams, searchParams)
# 进行KNN匹配
matches = flann.knnMatch(des1, des2, k = 2)

# 根据匹配的个数准备一个空的掩码数组（列表），用以绘制良好匹配的项
matchesMask = [[0, 0] for i in xrange(len(matches))]

# 枚举所有匹配项，记录距离比<0.7的掩码值
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]

# 以字典形式定义绘制参数
#   绘制匹配项的颜色: 绿色
#   单点颜色: 红色
#   匹配掩码: matchesMask
#   标志: 0
drawParams = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
# 绘制KNN匹配结果图像
resultImage = cv2.drawMatchesKnn(queryImage, kp1, trainingImage, kp2, matches, None, **drawParams)
# 显示结果图像
plt.imshow(resultImage, ), plt.show()