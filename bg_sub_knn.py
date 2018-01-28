# coding=utf-8
import cv2
import numpy as np
# 创建基于KNN的背景分割器
knn = cv2.createBackgroundSubtractorKNN(detectShadows = True)
# 获取10X10大小的椭圆形态结构化元素
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 12))
# 打开视频文件，以处理视频帧
camera = cv2.VideoCapture("movies/traffic.flv")

# 绘制检测到的移动目标轮廓矩形框
def drawCnt(fn, cnt):
    if cv2.contourArea(cnt) > 1400:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(fn, (x, y), (x + w, y + h), (255, 255, 0), 2)
# 循环处理每一帧
while True:
    # 读取一帧
    ret, frame = camera.read()
    # 若读取失败，退出循环
    if not ret:
        break
    # 应用KNN背景分割操作，返回前景图像
    fg = knn.apply(frame.copy())
    # 前景转化为BGR彩色图像
    fg_bgr = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
    # 将前景图像和当前帧进行按位与操作，返回差分图像
    bw_and = cv2.bitwise_and(fg_bgr, frame)
    # 将差分图像转化为灰度图
    draw = cv2.cvtColor(bw_and, cv2.COLOR_BGR2GRAY)
    # 高斯模糊
    draw = cv2.GaussianBlur(draw, (21, 21), 0)
    # 二值化
    draw = cv2.threshold(draw, 10, 255, cv2.THRESH_BINARY)[1]
    # 膨胀处理
    draw = cv2.dilate(draw, es, iterations = 2)
    # 查找轮廓
    image, contours, hierarchy = cv2.findContours(draw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制所有轮廓对象的矩形框
    for c in contours:
        drawCnt(frame, c)
    # 显示检测结果
    cv2.imshow("motion detection", frame)
    # 按Q键退出
    if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()