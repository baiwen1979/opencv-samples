# coding=utf-8
import cv2
import numpy as np
from os.path import join

# 训练图像的基础路径
datapath = "images/TrainImages/"
# 函数：根据训练图片的类名cls和编号i，返回图片的完整路径
def path(cls, i):
    image_path = "%s/%s%d.pgm"  % (datapath, cls, i + 1)
    print image_path
    return image_path

# 训练图片的类名：正例和反例
pos, neg = "pos-", "neg-"

# 创建SIFT对象，用于检测关键点
detect = cv2.xfeatures2d.SIFT_create()
# 创建SIFT对象，用于提取特征
extract = cv2.xfeatures2d.SIFT_create()

# 创建基于FLANN的匹配器实例
#   算法: algorithm = 1(FLANN_INDEX_KDTREE)
#   树个数: trees = 5
flann_params = dict(algorithm = 1, trees = 5)
matcher = cv2.FlannBasedMatcher(flann_params, {})

# 创建BOW训练器
#   簇数: 40
bow_kmeans_trainer = cv2.BOWKMeansTrainer(40)
# 初始化BOW图像特征提取器
extract_bow = cv2.BOWImgDescriptorExtractor(extract, matcher)

# 函数：从指定图像文件fn中提取SIFT特征
def extract_sift(fn):
    im = cv2.imread(fn, 0)
    return extract.compute(im, detect.detect(im))[1]
# 每个类都从训练数据集中读取8张图像，8个正样本，8个负样本
for i in range(8):
    bow_kmeans_trainer.add(extract_sift(path(pos, i)))
    bow_kmeans_trainer.add(extract_sift(path(neg, i)))
# 调用训练器对象的聚类函数，进行k-means聚类操作，并返回视觉词汇
voc = bow_kmeans_trainer.cluster()
# 将上述视觉词汇作为BOW图像特征提取器的视觉词汇
extract_bow.setVocabulary(voc)

# 函数：计算指定图像文件fn的（基于BOW的描述符提取器）的特征描述符
def bow_features(fn):
    im = cv2.imread(fn,0)
    return extract_bow.compute(im, detect.detect(im))

# 创建两个数组，分别对应训练数据和标签
traindata, trainlabels = [],[]
# 使用BOW图像特征提取器分别提取20幅正负样本图像的特征，并填充训练数据数组及相应的标签数组
# 1表示正匹配， -1表示负匹配
for i in range(20): 
    traindata.extend(bow_features(path(pos, i))); trainlabels.append(1)
    traindata.extend(bow_features(path(neg, i))); trainlabels.append(-1)

# 创建支持向量机
svm = cv2.ml.SVM_create()
# 将训练数据和标签放到NumPy数组，使用支持向量机进行训练
svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))

# 函数：使用训练好的支持向量机对图像文件fn进行预测，并输出预测结果
def predict(fn):
    f = bow_features(fn);  
    p = svm.predict(f)
    print fn, "\t", p[1][0][0]
    return p

# 使用上述函数对两幅图片（车和非车）进行预测
car, notcar = "images/car.jpg", "images/children.jpg"
car_img = cv2.imread(car)
notcar_img = cv2.imread(notcar)
car_predict = predict(car)
not_car_predict = predict(notcar)

# 指定用来在图像上显示预测结果（文字说明）的字体
font = cv2.FONT_HERSHEY_SIMPLEX
# 预测（检测）到汽车
if (car_predict[1][0][0] == 1.0):
    cv2.putText(car_img, 'Car Detected', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
# 未预测（检测）到汽车
if (not_car_predict[1][0][0] == -1.0):
    cv2.putText(notcar_img, 'Car Not Detected', (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
# 显示预测（检测）结果
cv2.imshow('BOW + SVM Success', car_img)
cv2.imshow('BOW + SVM Failure', notcar_img)
cv2.waitKey(0)
cv2.destroyAllWindows()