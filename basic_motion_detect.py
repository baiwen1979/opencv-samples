# coding=utf-8
import cv2
import numpy as np

# 打开系统默认摄像头
camera = cv2.VideoCapture(0)
# 获取10X10大小的椭圆形态结构化元素
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
# 5X5卷积核，即权值矩阵
kernel = np.ones((5, 5), np.uint8)
# 背景
background = None

while (True):
    # 读取一帧图像    
    ret, frame = camera.read()
    # 将第一帧作为整个输入的背景
    if background is None:
        # 对背景帧进行预处理，转化为灰阶图像
        background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 用高斯模糊对背景帧进行平滑处理
        background = cv2.GaussianBlur(background, (21, 21), 0)
        continue
    # 第一帧之后的其余所有帧：
    # 1. 灰度处理
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 2. 高斯模糊
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    # 3. 计算当前帧和背景帧的差异，得到差分图
    diff = cv2.absdiff(background, gray_frame)
    # 4. 对差分图应用阈值进行二值化处理，得到黑白图像
    diff = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)[1]
    # 5. 对二值化后的黑白图像进行膨胀(dilate)处理，从而对孔(hole)和缺陷(imperfection)进行归一化处理
    diff = cv2.dilate(diff, es, iterations = 2)
    # 6. 查找差分图像中（移动）目标的轮廓
    image, cnts, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制面积大于1500的轮廓边框
    for c in cnts:
        if cv2.contourArea(c) < 1500:
            continue
        # 计算轮廓C的矩形边框
        (x, y, w, h) = cv2.boundingRect(c)
        # 在当前帧中绘制矩形边框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    # 显示移动目标的矩形边框
    cv2.imshow("contours", frame)
    # 显示差分图
    cv2.imshow("diff", diff)
    # 按Q键退出
    if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
        break

cv2.destroyAllWindows()
camera.release()