# coding=utf-8
import cv2
# 样本生成程序：从摄像头视频中检测人脸，并生成用于人脸识别的样本图像
def generate():
    # 创建用于检测人脸的Haar分类器
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    # 创建用于检测眼睛的Haar分类器
    eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
    # 初始化摄像头
    camera = cv2.VideoCapture(0)
    # 用于生成样本图像编号的计数器变量
    count = 0
    while (True):
        # 读取视频帧
        ret, frame = camera.read()
        # 转化为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 人脸检测
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # 对于检测到每个人脸矩形
        for (x, y, w, h) in faces:
            # 绘制人脸矩形框
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # 扣取矩形人脸图像，并调整为200X200的大小
            f = cv2.resize(gray[y: y + h, x: x + w], (200, 200))
            # 将扣取出的人脸图像存储为PGM样本图像文件，用于训练
            cv2.imwrite('./images/%s.pgm' % str(count), f)
            # 输出计数器的值
            print count
            # 增加计数
            count += 1
        # 实时显示视频帧
        cv2.imshow("camera", frame)
        # 按Q键退出
        if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
            break
    # 释放摄像头并关闭所有窗口
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    generate()