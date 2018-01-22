# coding=utf-8
import cv2

def detect():
    # 创建用于人脸检测的Haar级联分类器
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    # 创建用于眼睛检测的Haar级联分类器
    eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
    # 初始化摄像头
    camera = cv2.VideoCapture(0)
    while (True):
        # 读取视频帧
        ret, frame = camera.read()
        # 将视频帧转化为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 人脸检测
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # 对于检测到的每个人脸矩形：
        for (x, y, w, h) in faces:
            # 在原始帧中绘制人脸矩形框（蓝色）
            img = cv2.rectangle(frame, (x, y), (x + w,y + h), (255, 0, 0), 2)
            # 从灰度图像中获取人脸矩形区域
            roi_gray = gray[y: y + h, x: x + w]
            # 在人脸矩形区域中进行人眼检测
            #    缩放因子：scaleFactor = 1.03
            #    最小近邻数：minNeighbors = 5
            #    标记：flags = 0
            #    最小搜索尺寸：minSize = (40, 40)，用于去掉所有假阳性
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, 0, (40, 40))
            # 对于检测到的每个人眼矩形
            for (ex, ey, ew, eh) in eyes:
                # 绘制人眼矩形框（绿色）
                cv2.rectangle(img, (x + ex, y + ey),(x + ex + ew,y + ey + eh), (0, 255, 0), 2)
        # 显示结果
        cv2.imshow("Camera Face Detection", frame)
        # 按Q键退出
        if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
            break
    # 释放摄像头
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect()