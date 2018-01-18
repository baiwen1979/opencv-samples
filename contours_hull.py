# coding=utf-8
import cv2
import numpy as np
# 加载图像
img = cv2.imread("images/hammer.jpg")
# 在源图像的灰度图像上执行一个二值化操作
ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) , 127, 255, cv2.THRESH_BINARY)
# 创建和源图像同样大小的黑色RGB图像
black = cv2.cvtColor(np.zeros((img.shape[1], img.shape[0]), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
# 轮廓检测
image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 对所有的源轮廓，执行以下操作
for cnt in contours:
  epsilon = 0.01 * cv2.arcLength(cnt, True) # 计算源轮廓c和近视多边形周长的最大差值
  approx = cv2.approxPolyDP(cnt,epsilon, True) # 计算源轮廓c的近似多边形框（闭合）
  hull = cv2.convexHull(cnt) # 计算源轮廓的凸形状(外壳)
  cv2.drawContours(black, [cnt], -1, (0, 255, 0), 2) # 在黑色RGB图像上绘制源轮廓，绿色，2像素粗
  cv2.drawContours(black, [approx], -1, (255, 0, 0), 2) # 在黑色RGB图像上绘制近似多边形框，蓝色
  cv2.drawContours(black, [hull], -1, (0, 0, 255), 2) # 在黑色RGB图像上绘制凸形状，红色

cv2.imshow("hull", black)
cv2.waitKey()
cv2.destroyAllWindows()