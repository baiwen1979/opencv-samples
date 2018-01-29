# coding=utf-8
import numpy as np
import cv2

# 从默认摄像头捕获视频帧
cap = cv2.VideoCapture('movies/traffic.flv')

# 捕获第一帧
ret,frame = cap.read()

# 设置跟踪窗口的初始位置
r, h, c, w = 300, 200, 400, 300
# 打包为元组
track_window = (c,r,w,h)

# 从当前帧中提取出ROI以进行跟踪
roi = frame[r: r + h, c: c + w]
# 将感兴趣区域（ROI）中的图像转换到HSV色彩空间
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# 创建一个包含具有HSV值的ROI所有像素的掩码，HSV值在上界与下界之间
mask = cv2.inRange(hsv_roi, np.array((100., 30. ,32.)), np.array((180., 120., 255.)))
# 计算ROI的色彩直方图
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])
# 将直方图的值归一化到0～255范围内
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
# 均值漂移在达到收敛之前会迭代多次，但不能保证一定收敛。
# 因此，这里要指定停止条件(Termination Criteria)：
# 均值漂移迭代10次后或者中心移动至少1个像素时，均值漂移就停止计算中心移动
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while True:
    # 读取一帧
    ret ,frame = cap.read()
    # 若成功读取一帧
    if ret == True:
        # 转化到HSV色彩空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 对HSV色彩空间图像数组执行直方图反向投影
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        ## 应用连续自适应均值漂移（CAMShift）以获取新的跟踪窗口和位置
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        ## 找到返回的被旋转矩形的顶点
        pts = cv2.boxPoints(ret)
        ## 顶点坐标值转化为整数
        pts = np.int0(pts)
        ## 使用折线函数在帧上绘制矩形线段
        img2 = cv2.polylines(frame, [pts], True, 255, 2)

        # 显示帧
        cv2.imshow('img2', img2)
        # 按ESC退出
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
