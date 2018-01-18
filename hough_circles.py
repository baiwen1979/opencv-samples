# coding=utf-8
import cv2
import numpy as np

# 加载图像
planets = cv2.imread('images/planet_glow.jpg')
# 转为灰度图像
gray_img = cv2.cvtColor(planets, cv2.COLOR_BGR2GRAY)
# 中值模糊
img = cv2.medianBlur(gray_img, 5)
# 转为RGB彩色图像
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# 检测圆
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 120,
                            param1 = 100, param2 = 30, minRadius = 0, maxRadius = 0)
# 转换为整数值
circles = np.uint16(np.around(circles))

# 绘制所有检测出的圆
for i in circles[0, :]:
    # 绘制外部圆
    cv2.circle(planets, (i[0],i[1]), i[2], (0,255,0), 2)
    # 绘制圆心
    cv2.circle(planets, (i[0],i[1]), 2, (0,0,255), 3)

cv2.imwrite("images/planets_circles.jpg", planets)
cv2.imshow("HoughCirlces", planets)
cv2.waitKey()
cv2.destroyAllWindows()