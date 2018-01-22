# coding=utf-8
import cv2

filename = 'images/children.jpg'

# 图片文件中人脸检测程序
def detect(filename):
    # 创建用于人脸检测的Haar级联分类器
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    # 创建用于眼睛检测的Haar级联分类器
    eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
    # 加载包含人脸的图片
    img = cv2.imread(filename)
    # 将图片转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 人脸检测：缩放因子(scaleFactor): 1.3
    #         最小紧邻数(minNeighbors): 5
    faces = face_cascade.detectMultiScale(gray, 1.2, 8)
    # 对于检测出的每个人脸矩形
    for (x, y, w, h) in faces:
        # 在原图上绘制人脸蓝色矩形
        img = cv2.rectangle(img,(x, y),(x + w, y + h),(255, 0, 0), 2)
    # 显示并保存检测结果
    cv2.namedWindow('Faces Detected!!')
    cv2.imshow('Faces Detected!!', img)
    cv2.imwrite('images/faces_detected.jpg', img)
    cv2.waitKey(0)

detect(filename)