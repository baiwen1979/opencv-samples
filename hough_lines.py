#coding=utf8
import cv2
import numpy as np

img = cv2.imread('images/lines.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 使用边缘检测滤波器转为单通道二值图像
edges = cv2.Canny(gray, 50, 120)
# 设置最小直线长度，小于此值的线段会被消除
minLineLength = 15
# 最大线段间隔，大于此值的线段会被分开
maxLineGap = 5
# 直线检测
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength, maxLineGap)
# 绘制直线
for x1, y1, x2, y2 in lines[0]:
    cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)

cv2.imshow("edges", edges)
cv2.imshow("lines", img)
cv2.waitKey()
cv2.destroyAllWindows()