# coding=utf-8
import cv2
import numpy as np

# 加载图像（下采样）
img = cv2.pyrDown(cv2.imread("images/hammer.jpg"), cv2.IMREAD_UNCHANGED)

# 在源图像的灰度图像上执行一个二值化操作
ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 
                            127, 255, cv2.THRESH_BINARY)
# 轮廓检测，返回修改后的图像，图像的所有轮廓和层级
image, contours, hier = cv2.findContours(thresh, 
                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 对所有的轮廓c执行以下操作
for c in contours:
    # 获得轮廓c的边界框
    x, y, w, h = cv2.boundingRect(c)
    # 绘制红色的边界框矩形，粗细为2像素
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 计算包围轮廓c的最小矩形区域
    rect = cv2.minAreaRect(c)
    # 计算最小矩形区域顶点的坐标
    box = cv2.boxPoints(rect)
    # 将顶点坐标转化为整数
    box = np.int0(box)
    # 绘制此矩形（由最小矩形区域顶点构成的轮廓），红色，3像素粗
    cv2.drawContours(img, [box], 0, (0, 0, 255), 3)

    # 计算轮廓c最小闭圆的圆心坐标和半径
    (x, y), radius = cv2.minEnclosingCircle(c)
    # 转化为整数
    center = (int(x), int(y))
    radius = int(radius)
    # 绘制此圆形，绿色，2像素粗
    img = cv2.circle(img, center, radius, (0, 255, 0), 2)

# 绘制所有轮廓，蓝色，1像素粗
cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

cv2.imshow("contours", img)
cv2.waitKey()
cv2.destroyAllWindows()
