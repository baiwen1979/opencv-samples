# coding=utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt
# 打开默认摄像头
camera = cv2.VideoCapture(0)

while True:
    # 读取一帧
    ret, img = camera.read()
    # 颜色三元组
    color = ('b','g','r')
    # 统计并绘制三元色的直方图
    for i, col in enumerate(color):
        # 计算图像帧数组的色彩直方图
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        # 显示直方图
        plt.plot(histr, color = col)
        plt.xlim([0, 256])
        plt.show()

camera.release()
cv2.destroyAllWindows()