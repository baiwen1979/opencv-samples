# coding=utf-8
import numpy as np
import cv2
from matplotlib import pyplot as plt
# 最小匹配次数
MIN_MATCH_COUNT = 10
# 加载原目标(seed)图片
img1 = cv2.imread('images/bb.jpg', 0)
# 加载包含原目标内容的图片
img2 = cv2.imread('images/color2_small.jpg', 0)

# 初始化 SIFT 检测器
sift = cv2.xfeatures2d.SIFT_create()

# 使用SIFT检测关键点并计算描述符
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 使用FLANN索引树算法
FLANN_INDEX_KDTREE = 0
# 设置FLANN匹配器参数（字典）
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
# 创建FLANN匹配器
flann = cv2.FlannBasedMatcher(index_params, search_params)
# 先进行KNN匹配，确保一定数目的良好匹配，计算单应性至少需要4个匹配
matches = flann.knnMatch(des1, des2, k = 2)

# 存储良好（距离比<0.7) 的匹配（到goodMatches数组）
goodMatches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        goodMatches.append(m)
#   如果良好匹配的个数大于指定的最小匹配次数
if len(goodMatches) > MIN_MATCH_COUNT:
    # 分别创建记录原始图像和训练图像中良好匹配的关键点坐标（位置）的数组
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in goodMatches ])
    print src_pts.shape
    src_pts = src_pts.reshape(-1, 1, 2)
    print src_pts.shape
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in goodMatches ])
    print dst_pts.shape
    dst_pts = dst_pts.reshape(-1, 1, 2)
    print dst_pts.shape
    # 进行单应性匹配，返回匹配项掩码(mask)和畸变矩阵(M)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # 用来绘制匹配图的匹配项掩码
    matchesMask = mask.ravel().tolist()
    # 获得原目标图像的高度和宽度
    h, w = img1.shape
    # 对匹配目标（第2幅）图像计算相对于原目标图像的投影畸变
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    # 绘制畸变边框
    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

else:
    print "Not enough matches are found - %d/%d" % (len(goodMatches), MIN_MATCH_COUNT)
    matchesMask = None
# 设置绘制参数
draw_params = dict(matchColor = (0, 255, 0), # 将匹配项绘制为绿色
                   singlePointColor = None, # 不绘制其它点
                   matchesMask = matchesMask, # 只绘制匹配项
                   flags = 2)
# 进行绘制
img3 = cv2.drawMatches(img1, kp1, img2, kp2, goodMatches, None, **draw_params)

plt.imshow(img3, 'gray'),plt.show()