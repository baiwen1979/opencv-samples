# coding=utf-8
import os
import sys
import cv2
import numpy as np

# 归一化数组X的元素到[low, high]
def normalize(X, low, high, dtype=None):
    """Normalizes a given array in X to a value between low and high."""
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # 归一化到 [0...1]
    X = X - float(minX)
    X = X / float((maxX - minX))
    # 缩放到 [low...high]
    X = X * (high-low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)

# 加载指定目录下的所有图像
def read_images(path, sz = None):
    """读取指定目录下的所有图像，若指定参数sz（大小），则重新设置图像的大小
    参数:
        path: 图像所在的目录
        sz: 图像大小（二元组）
    返回:
        列表 [X,y]
            X: 图像数组（列表）
            y: 图像数组X所对应的标签
    """
    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    if (filename == ".directory"):
                        continue
                    filepath = os.path.join(subject_path, filename)
                    im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    if (im is None):
                        print "image " + filepath + " is none"
                    else:
                        print subdirname, "=>", filepath
                    # 调整大小，如果指定的话
                    if (sz is not None):
                        im = cv2.resize(im, (200, 200))
                    X.append(np.asarray(im, dtype = np.uint8))
                    y.append(int(subdirname))
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
    print y
    return [X,y]

# 识别摄像头中的人脸
def face_rec():
    names = ['Unknown', 'Dad', 'Xiner', 'Mam']
    if len(sys.argv) < 2:
        print "USAGE: face_rec.py </path/to/images> [</path/to/store/images/at>]"
        sys.exit()

    [X,y] = read_images(sys.argv[1])
    y = np.asarray(y, dtype = np.int32)
    
    if len(sys.argv) == 3:
        out_dir = sys.argv[2]
    # 创建Eigenfaces模型，使用默认参数
    # model = cv2.face.EigenFaceRecognizer_create()
    # 或：创建Fisherfaces模型
    # model = cv2.face.FisherFaceRecognizer_create()
    # 或：创建LBPH模型
    model = cv2.face.LBPHFaceRecognizer_create()
    # 以图像数组和标签数组为参数，训练模型
    model.train(np.asarray(X), np.asarray(y))
    # 初始化摄像头
    camera = cv2.VideoCapture(0)
    # 创建Haar人脸检测分类器
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    while (True):
        # 读取视频帧    
        read, img = camera.read()
        # 检测人脸
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        # 对检测到的所有人脸
        for (x, y, w, h) in faces:
            # 绘制人脸矩形框
            img = cv2.rectangle(img,(x, y),(x + w,y + h),(255, 0, 0), 2)
            # 转化为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 扣取要识别的人脸图像
            roi = gray[x: x + w, y: y + h]
            try:
                # 调整人脸图像大小为200X200
                roi = cv2.resize(roi, (200, 200), interpolation = cv2.INTER_LINEAR)
                # 输出大小
                print roi.shape
                # 使用训练好的模型进行人脸识别
                params = model.predict(roi)
                # 输出识别结果：标签和置信度评分（所识别人脸和模型的差距，越小越匹配）
                print "Label: %s, Confidence: %.2f" % (params[0], params[1])
                # 在图像中输出标签文本
                cv2.putText(img, names[params[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                if (params[0] == 1):
                    cv2.imwrite('face_rec.jpg', img)
            except:
                continue
        # 显示视频窗口
        cv2.imshow("camera", img)
        # 按Q键退出
        if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
            break
    # 释放摄像头并关闭所有窗口
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_rec()