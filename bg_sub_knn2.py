# coding=utf-8
import cv2
import numpy as np
# 创建基于KNN的背景分割器
bs = cv2.createBackgroundSubtractorKNN(detectShadows = True)
# 获取形态结构元素(3X3椭圆)
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# 捕获视频文件中的帧
vc = cv2.VideoCapture('movies/traffic.flv')
# 循环处理每一帧
while True:
    # 读取一帧
    ret, frame = vc.read()
    # 对当前帧进行背景分割，返回前景掩码
    fgmask = bs.apply(frame)
    # 二值化
    th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
    # 膨胀
    dilated = cv2.dilate(th, es, iterations = 2)
    # 轮廓检测
    image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓的矩形边框
    for c in contours:
        # 只绘制面积大于1600的轮廓的矩形边框
        if cv2.contourArea(c) > 1600:
            # 获得轮廓的矩形边框的位置和大小
            (x, y, w, h) = cv2.boundingRect(c)
            # 在原始帧上绘制矩形框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    
    # 显示前景掩码图像
    cv2.imshow('foreground mask', fgmask)
    # 显示二值化后的黑白图像
    cv2.imshow('threshold image', th)
    # 显示目标检测结果
    cv2.imshow('detection frames', frame)
    # 按ESC退出
    if cv2.waitKey(10) & 0xff == 27:
        break
# 关闭视频
vc.release()
# 销毁所有窗口
cv2.destroyAllWindows()
            


