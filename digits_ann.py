# coding=utf-8
import cv2
import cPickle
import numpy as np
import gzip

"""
OpenCV ANN 手写数字识别示例
该库封装了OpenCV的ANN，实现了数据的自动加载并提供了默认参数，
比如20个隐藏层，10000个样本和1次训练
加载数据的代码摘自 http://neuralnetworksanddeeplearning.com/chap1.html
由Michael Nielsen提供
"""
# 函数：加载数据
def load_data():
    # 打开包含训练数据和测试数据的数据（压缩）文件
    mnist = gzip.open('data/mnist.pkl.gz', 'rb')
    # 使用pickle库从打开的序列化数据文件中加载训练数据，分类标签和测试数据
    training_data, classification_data, test_data = cPickle.load(mnist)
    # 关闭数据文件
    mnist.close()
    # 以三元组的形式返回训练数据，分类标签和测试数据
    return (training_data, classification_data, test_data)

# 函数：包装数据
def wrap_data():
    # 加载训练数据，分类（验证）数据和测试数据
    tr_d, va_d, te_d = load_data()
    # 包装训练数据
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    # 包装验证数据
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    # 包装测试数据
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    # 返回包装数据的三元组
    return (training_data, validation_data, test_data)

# 函数：向量化，返回整数j对应的向量
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

# 函数：创建ANN,用于手写数字识别
def create_ANN(hidden = 20):
    # 创建ANN多层感知器
    ann = cv2.ml.ANN_MLP_create()
    # 设置各层的神经元个数，其中输入层为784，隐藏层默认为20，输出层为10
    ann.setLayerSizes(np.array([784, hidden, 10]))
    # 设置训练方法为弹性反馈
    ann.setTrainMethod(cv2.ml.ANN_MLP_RPROP)
    # 设置激活函数为Sigmoid
    ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
    # 设置停止条件：迭代100次
    ann.setTermCriteria(( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1 ))
    return ann

# 函数：使用指定的训练样本数(samples)和训练周期数(epochs)对指定的ANN(ann)进行训练
def train(ann, samples = 10000, epochs = 1):
    # 加载包装好的训练数据，验证（标签）数据和测试数据
    tr, val, test = wrap_data()
    # 按指定的周期数进行训练   
    for x in xrange(epochs):
        # 训练样本计数
        counter = 0
        # 对每个手写数字样本图片进行训练
        for img in tr: 
            # 训练完成？      
            if (counter > samples):
                break
            # 每训练1000个样本，输出训练周期，已训练样本数和样本总数
            if (counter % 1000 == 0):
                print "Epoch %d: Trained %d/%d" % (x, counter, samples)
            # 增加计数
            counter += 1
            # 训练数据（图像）和相应的分类标签（数字）
            data, digit = img
            # 将训练数据和分类标签转化为一维数组（ANN期望的训练数据格式）交给ANN进行训练
            ann.train(np.array([data.ravel()], dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array([digit.ravel()], dtype=np.float32))
        # 一个训练周期完成
        print "Epoch %d complete" % x
    # 返回训练好的ANN及测试数据
    return ann, test

# 函数：使用指定的测试数据(test_data)对训练好的ANN(ann)进行测试 
def test(ann, test_data):
    # 将测试样本数据转化为28X28图片（矩阵）
    sample = np.array(test_data[0][0].ravel(), dtype=np.float32).reshape(28, 28)
    # 显示测试样本图片
    cv2.imshow("sample", sample)
    cv2.waitKey()
    # 输出预测结果
    print ann.predict(np.array([test_data[0][0].ravel()], dtype=np.float32))

# 函数：使用训练好的ANN(ann)对样本图像(sample)进行预测
def predict(ann, sample):
    # 获取样本图像的副本
    resized = sample.copy()
    # 获得样本图像的形状（高度和宽度）
    rows, cols = resized.shape
    # 检测测试样本图像的形状与预期是否相符
    if (rows != 28 or cols != 28) and rows * cols > 0:
        # 若不相符，则重新调整
        resized = cv2.resize(resized, (28, 28), interpolation = cv2.INTER_LINEAR)
    # 预测并返回分类结果
    return ann.predict(np.array([resized.ravel()], dtype=np.float32))

"""
用法:
ann, test_data = train(create_ANN())
test(ann, test_data)
"""