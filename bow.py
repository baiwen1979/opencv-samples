# coding=utf-8
import cv2
import numpy as np
from os.path import join

# 函数：根据分类名称和索引获得图片的完整路径
def path(cls, i):
    return "%s/%s%02d.jpg"  % (datapath, cls, i + 1)

# 创建用于特征检测的SIFT实例
detect = cv2.xfeatures2d.SIFT_create()
# 创建用于特征提取的SIFT实例
extract = cv2.xfeatures2d.SIFT_create()

# 设置参数并创建FLANN匹配器
flann_params = dict(algorithm = 1, trees = 5)      # 设置匹配器参数，算法：FLANN_INDEX_KDTREE=1
matcher = cv2.FlannBasedMatcher(flann_params, {})  # 第二个参数为空字典

# 创建k-means词袋（BOW）训练器
bow_train   = cv2.BOWKMeansTrainer(8)
# 以上述SIFT特征提取器和FLANN匹配器为参数创建BOW图像描述符提取器
bow_extract = cv2.BOWImgDescriptorExtractor( extract, matcher )

#函数：从指定图像文件fn中提取SIFT特征 
def feature_sift(fn):
    """
    从指定的图像文件中提取SIFT特征

    @param: fn 图像文件的路径
    
    @returns: 从图像文件中提取出的SIFT特征
    """
    im = cv2.imread(fn, 0)
    return extract.compute(im, detect.detect(im))[1]

# 训练图像的基础路径
basepath = "images/"
# 训练图像列表
images = ["children.jpg", "beans.jpg", "basil.jpg", "hammer.jpg"]

# 将训练图像的SIFT特征添加到BOW K-Means（词袋K均值）训练器中，准备训练
for i in range(len(images)):
    print images[i]
    bow_train.add(feature_sift(join(basepath, images[i])))

# 调用BOW训练器的聚类函数进行K-Means分类，得到视觉词汇（Vocabulary)
voc = bow_train.cluster()
# 将视觉词汇作为BOW特征提取器的词汇参数
bow_extract.setVocabulary(voc)
# 输出BOW视觉词汇
print "bow vocab", np.shape(voc), voc

# 函数：计算指定图像文件fn的（基于BOW的描述符提取器）BOW特征描述符
def feature_bow(fn):
    im = cv2.imread(fn,0)
    return bow_extract.compute(im, detect.detect(im))

# 创建两个数组，分别对应训练数据和标签
traindata, trainlabels = [],[]

# 使用上述BOW特征提取函数计算训练图像的特征，并填充训练数据数组及相应的标签数组
traindata.extend(feature_bow(join(basepath, images[0])))
traindata.extend(feature_bow(join(basepath, images[1])))
traindata.extend(feature_bow(join(basepath, images[2])))
traindata.extend(feature_bow(join(basepath, images[3])))
# 第一幅图像作为正例
trainlabels.append(1)
# 其余三幅图像作为反例
trainlabels.append(-1)
trainlabels.append(-1)
trainlabels.append(-1)
# 输出训练数据及其样本大小
print "svm items", len(traindata), len(traindata[0])

# 创建SVM支持向量机
svm = cv2.ml.SVM_create()
# 将训练数据和标签放到NumPy数组，使用支持向量机进行训练
svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))

# 函数： 使用训练好的支持向量机SVM对图像文件fn的分类进行预测
def predict(fn):
    f = feature_bow(fn);  
    p = svm.predict(f)
    print fn, "\t", p[1][0][0] 

# 预测图片bb.jpg的分类
for i in range(len(images)):
    predict(join(basepath, images[i]))