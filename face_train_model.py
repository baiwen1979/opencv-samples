#coding=utf-8
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
    X = X * (high - low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)

# 加载目录下的所有图像
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
    # 遍历目录
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    if (filename == ".directory"):
                        continue
                    filepath = os.path.join(subject_path, filename)
                    # 读取图像
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    if (im is None):
                        print "image " + filepath + " is none" 
                    # 调整大小，如果指定的话
                    if (sz is not None):
                        im = cv2.resize(im, sz)
                    # 将图像以二维数组的形式添加到X数组（列表）中
                    X.append(np.asarray(im, dtype = np.uint8))
                    # 为X添加标签
                    y.append(int(subdirname))
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
    return [X,y]

if __name__ == "__main__":
    # 图像的输出目录
    out_dir = None
    # 要求在命令行参数中指定图像的输入和输出目录
    if len(sys.argv) < 2:
        print "USAGE: face_train_model.py </path/to/images> [</path/to/store/images/at>]"
        sys.exit()
    # 从输入目录中（第一个命令行参数）读取图像到[X,y]数组中
    [X, y] = read_images(sys.argv[1])
    # 将标签转化为32位整数
    y = np.asarray(y, dtype = np.int32)
    # 将第二个命令行参数作为输出目录
    if len(sys.argv) == 3:
        out_dir = sys.argv[2]
    # 创建Eigenfaces模型，使用默认参数
    model = cv2.face.EigenFaceRecognizer_create()
    # 以图像数组和标签数组为参数，训练模型
    model.train(np.asarray(X), np.asarray(y))
    # 采用训练好的模型对图像数组中的第一幅（人脸）图像进行识别（标注）
    [p_label, p_confidence] = model.predict(np.asarray(X[80]))
    # 输出检测（预测）结果: 标签和置信度
    print "Predicted label = %d (confidence = %.2f)" % (p_label, p_confidence)
    # 输出模型参数:
    print model.getLabels()
    # 获取均值矩阵:
    mean = model.getMean()
    # 获取特征向量
    eigenvectors = model.getEigenVectors()
    # 存储均值矩阵前，先对其归一化到[0, 255],便于以图像的形式存储:
    mean_norm = normalize(mean, 0, 255, dtype = np.uint8)
    # 将归一化后的均值矩阵重新调整大小，使其和图像一样大
    mean_resized = mean_norm.reshape(X[0].shape)
    # 如果没指定输出目录，则显示之，否则存储到输出目录
    if out_dir is None:
        cv2.imshow("mean", mean_resized)
    else:
        cv2.imwrite("%s/mean.png" % (out_dir), mean_resized)
    # 将前16（最多）个特征向量转化为灰度图像
    # 注：特征向量以列存储
    for i in xrange(min(len(X), 16)):
        eigenvector_i = eigenvectors[:,i].reshape(X[0].shape)
        eigenvector_i_norm = normalize(eigenvector_i, 0, 255, dtype=np.uint8)
        # 显示或存储人脸特征图像:
        if out_dir is None:
            cv2.imshow("%s/eigenface_%d" % (out_dir,i), eigenvector_i_norm)
        else:
            cv2.imwrite("%s/eigenface_%d.png" % (out_dir,i), eigenvector_i_norm)
    # 显示结果:
    if out_dir is None:
        cv2.waitKey(0)