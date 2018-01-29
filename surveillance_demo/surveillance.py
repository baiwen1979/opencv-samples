#! /usr/bin/python
# coding=utf-8

"""
行人监控示例: 跟踪摄像头中的行人
该程序打开视频（摄像头或视频文件）并跟踪视频中的行人
"""
# 导入模块
import cv2
import numpy as np
import os.path as path
import argparse

# 命令行参数解析器
parser = argparse.ArgumentParser()
# 为参数解析器添加参数及其帮助信息
parser.add_argument("-a", "--algorithm",
    help = "m (or nothing) for meanShift and c for camshift")
# 解析命令行参数，并将解析结果转化为内部变量（字典）
args = vars(parser.parse_args())

# 函数：计算给定矩阵的中心
def center(points):
    """计算给定矩阵的中心"""
    x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
    y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
    return np.array([np.float32(x), np.float32(y)], np.float32)
# 图像上显示文字的字体
font = cv2.FONT_HERSHEY_SIMPLEX

# 类：行人
class Pedestrian():
    """
    行人类
    每个行人由ROI、ID和一个卡尔曼滤波器构成，此行人类用来保持这些状态
    """

    # 构造器函数：传递ID，当前视频帧以及跟踪窗口的坐标（包括宽和高）
    def __init__(self, id, frame, track_window):
        """使用跟踪的窗口坐标初始化行人对象"""
        # 设置行人的id
        self.id = int(id)
        # 记录并设置跟踪窗口的坐标和大小
        x, y, w, h = track_window
        self.track_window = track_window
        # 设置ROI，从当前帧中提取感兴趣区域
        self.roi = cv2.cvtColor(frame[y: y + h, x: x + w], cv2.COLOR_BGR2HSV)
        # 计算并设置给定ROI的直方图
        roi_hist = cv2.calcHist([self.roi], [0], None, [16], [0, 180])
        self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # 设置卡尔曼滤波器
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
        # 设置测量和预测坐标
        self.measurement = np.array((2, 1), np.float32) 
        self.prediction = np.zeros((2, 1), np.float32)
        # 设置终止条件
        self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        # 设置中心点
        self.center = None
        # 更新视频帧
        self.update(frame)

    # 析构函数 
    def __del__(self):
        print "Pedestrian %d destroyed" % self.id
    
    # 方法：更新视频帧frame中行人的信息，实现对当前行人的跟踪
    def update(self, frame):
        # 将当前视频帧转换到HSV色彩空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 计算行人HSV直方图的反向投影
        back_project = cv2.calcBackProject([hsv],[0], self.roi_hist, [0,180], 1)
        # 根据命令行参数选择漂移算法
        # 若命令行参数algorithm为c,则使用CAM漂移算法计算跟踪窗口，并绘制到当前帧
        if args.get("algorithm") == "c":
            # 使用CAM漂移算法计算跟踪窗口的坐标和大小
            ret, self.track_window = cv2.CamShift(back_project, self.track_window, self.term_crit)
            # 计算跟踪窗口的顶点坐标
            pts = cv2.boxPoints(ret)
            # 坐标值转化为整数
            pts = np.int0(pts)
            # 计算跟踪窗口的中心点，并设置为当前行人的中点属性
            self.center = center(pts)
            # 绘制跟踪框
            cv2.polylines(frame, [pts], True, 255, 1)
        # 若没有指定命令行参数，或命令行参数algorithm的值为m,则使用均值漂移算法
        if not args.get("algorithm") or args.get("algorithm") == "m":
            # 使用均值漂移算法计算跟踪窗口的坐标和大小
            ret, self.track_window = cv2.meanShift(back_project, self.track_window, self.term_crit)
            # 计算并设置当前行人的中点属性
            x, y, w, h = self.track_window
            self.center = center([[x, y],[x + w, y],[x, y + h],[x + w, y + h]]) 
            # 绘制跟踪框 
            cv2.rectangle(frame, (x,y), (x + w, y + h), (255, 255, 0), 2)
        # 使用卡尔曼滤波器校正中心点
        self.kalman.correct(self.center)
        # 使用卡尔曼滤波器预测行人位置
        prediction = self.kalman.predict()
        # 以预测点为圆心绘制圆
        cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 4, (255, 0, 0), -1)
        # 在当前帧中显示行人信息文本阴影
        cv2.putText(frame, "ID: %d -> %s" % (self.id, self.center), (11, (self.id + 1) * 25 + 1),
            font, 0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA)
        # 在当前帧中显示行人信息文本
        cv2.putText(frame, "ID: %d -> %s" % (self.id, self.center), (10, (self.id + 1) * 25),
            font, 0.6,
            (0, 255, 0),
            1,
            cv2.LINE_AA)

# 主函数
def main():
    # 打开视频
    # camera = cv2.VideoCapture(path.join(path.dirname(__file__), '../movies', "traffic.flv"))
    # camera = cv2.VideoCapture(path.join(path.dirname(__file__), '../movies', '768x576.avi'))
    # camera = cv2.VideoCapture(path.join(path.dirname(__file__), "../movies", "movie.mpg"))
    camera = cv2.VideoCapture(0)

    # 用于MOG背景分割器的历史帧数量
    history = 20
    # 使用KNN背景分割器
    bs = cv2.createBackgroundSubtractorKNN()

    # MOG 背景分割器
    # bs = cv2.bgsegm.createBackgroundSubtractorMOG(history = history)
    # bs.setHistory(history)

    # GMG 背景分割器
    # bs = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames = history)
    
    # 创建命名窗口
    cv2.namedWindow("surveillance")
    # 行人字典
    pedestrians = {}
    # 是否使用第一帧作为背景帧
    firstFrame = True
    # 帧计数器
    frames = 0
    # 使用XVID视频编码器，输出跟踪视频
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('movies/output.avi', fourcc, 20.0, (640,480))

    # 循环处理每一帧
    while True:
        # 输出当前帧序号
        print " -------------------- FRAME %d --------------------" % frames
        # 读取一帧
        grabbed, frame = camera.read()
        # 读取失败，退出
        if (grabbed is False):
            print "failed to grab frame."
            break
        # 分割背景帧，得到前景掩码
        fgmask = bs.apply(frame)

        # 帧计数器，用于背景分割器构建历史记录
        if frames < history:
            frames += 1
            continue

        # 二值化
        th = cv2.threshold(fgmask.copy(), 127, 255, cv2.THRESH_BINARY)[1]
        # 腐蚀
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 2)
        # 膨胀
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,3)), iterations = 2)
        # 查找轮廓
        image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        counter = 0
        # 绘制轮廓的边界矩形框：
        for c in contours:
            # 只绘制面积大于500的轮廓的矩形框
            if cv2.contourArea(c) > 500:
                # 计算轮廓的边界矩形框
                (x, y, w, h) = cv2.boundingRect(c)
                # 绘制矩形框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                # 只在首帧创建行人，然后进行跟踪
                if firstFrame is True:
                    # 创建行人，并以counter为ID添加到字典
                    pedestrians[counter] = Pedestrian(counter, frame, (x, y, w, h))
                    counter += 1
        
        # 对每个行人更新视频帧
        for i, p in pedestrians.iteritems():
            p.update(frame)
        # 更新首帧标志和帧计数器
        firstFrame = False
        frames += 1
        # 在命名窗口中显示当前帧
        cv2.imshow("surveillance", frame)
        # 并输出当前帧
        out.write(frame)
        # 按ESC退出
        if cv2.waitKey(110) & 0xff == 27:
            break
    # 关闭输出视频
    out.release()
    # 关闭输入视频
    camera.release()
# 主程序入口
if __name__ == "__main__":
  main()