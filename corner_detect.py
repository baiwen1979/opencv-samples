# coding=utf-8
import cv2
import numpy as np
import sys

img = cv2.imread('images/' + sys.argv[1])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
# 使用cornerHarris算法进行角点检测
# 角点块的大小: blockSize = 9
# Sobel算子中孔大小: kSize = 23
dst = cv2.cornerHarris(gray, 9, 23, 0.04)
# 将检测到的角点标记为红色
img[dst > 0.01 * dst.max()] = [0, 0, 255] 
while (True):
    cv2.imshow('corners', img)
    # 按Q键退出
    if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
        break
cv2.destroyAllWindows()