# coding=utf-8
import cv2
import numpy as np

# 训练图像的基本路径
datapath = "images/TrainImages/"
# 样本个数
SAMPLES = 400

# 函数：以图像类的名称(cls:neg-/pos-)作为前缀，返回第i幅图像的完整路径
def path(cls, i):
    '''
    以图像类的名称(cls:neg-/pos-)作为前缀,返回第i幅图像的完整路径
    '''
    return "%s/%s%d.pgm"  % (datapath, cls, i + 1)

# 函数：获取FLANN匹配器
def get_flann_matcher():
  flann_params = dict(algorithm = 1, trees = 5)
  return cv2.FlannBasedMatcher(flann_params, {})

# 函数：获取BOW图像特征提取器
def get_bow_extractor(extract, match):
    '''
    以指定的特性提取器(extract)和匹配器(match)为参数，创建并返回BOW图像特征（描述符）提取器
    '''
    return cv2.BOWImgDescriptorExtractor(extract, match)

# 函数：获取分别用于特征检测和特征提取的两个SIFT实例
def get_extract_detect():
    '''
    返回分别用于特征检测和特征提取的两个SIFT实例
    '''
    return cv2.xfeatures2d.SIFT_create(), cv2.xfeatures2d.SIFT_create()

# 函数：使用SIFT特征检测和特征提取器提取图像文件fn的特征
def extract_sift(fn, extractor, detector):
    '''
    使用SIFT特征检测器(extractor)和特征提取器(detector)提取图像文件(fn)的特征
    '''
    im = cv2.imread(fn, 0)
    return extractor.compute(im, detector.detect(im))[1]

# 函数：使用BOW特征提取器和SIFT特征检测器提取图像的特征   
def bow_features(img, extractor_bow, detector):
    '''
    使用BOW特征提取器(extractor_bow)和SIFT特征检测器(detector)提取图像(img的特征
    '''
    return extractor_bow.compute(img, detector.detect(img))

# 函数：汽车检测函数
def car_detector():
    # 正例和反例前缀
    pos, neg = "pos-", "neg-"

    # 获取特征检测和提取器实例
    detect, extract = get_extract_detect()
    # 获取FLANN匹配器对象
    matcher = get_flann_matcher()

    # 输出处理步骤1
    print "1. building BOWKMeansTrainer..."
    # 创建基于词袋（BOW）的K-Means训练器，K = 12
    bow_kmeans_trainer = cv2.BOWKMeansTrainer(12)
    # 获取BOW特征提取器
    extract_bow = get_bow_extractor(extract, matcher)

    # 输出处理步骤2
    print "2. adding features to trainer"
    # 将所有样本图像的特征添加到训练器中
    for i in range(SAMPLES):
        print i
        bow_kmeans_trainer.add(extract_sift(path(pos, i), extract, detect))
    # 调用BOW K-Means训练器的聚类函数生成样本图像的视觉词汇
    vocabulary = bow_kmeans_trainer.cluster()
    # 将视觉词汇提供给BOW特征提取器
    extract_bow.setVocabulary(vocabulary)

    # 定义数组，分别用来存储从训练图像中提取的BOW特征和相应的标签
    traindata, trainlabels = [],[]
    # 输出处理步骤 3
    print "3. adding to train data"
    # 使用样本图像的BOW特征填充训练数据数组及其标签（正例为1，反例为-1）
    # 从而将训练数据和类进行关联
    for i in range(SAMPLES):
        print i
        # 增加一个正样本，设置标签为1
        traindata.extend(bow_features(cv2.imread(path(pos, i), 0), extract_bow, detect))
        trainlabels.append(1)
        # 增加一个负样本，设置标签为-1
        traindata.extend(bow_features(cv2.imread(path(neg, i), 0), extract_bow, detect))
        trainlabels.append(-1)

    # 创建SVM支持向量机进行分类训练
    svm = cv2.ml.SVM_create()
    # 设置支持向量机的类型
    svm.setType(cv2.ml.SVM_C_SVC)
    # 设置Gamma值
    svm.setGamma(1)
    # 设置训练误差和预测误差
    svm.setC(35)
    # 设置分类器的核函数，这里使用SVM_RBF(基于高斯函数的Radial Basis Function)
    svm.setKernel(cv2.ml.SVM_RBF)
    # 使用训练数据及其标签进行训练
    svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))
    # 返回训练好的SVM和BOW提取器对象
    return svm, extract_bow