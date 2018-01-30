# coding=utf-8
import cv2
import numpy as np
from random import randint
# 创建ANN多层感知器神经网络
animals_net = cv2.ml.ANN_MLP_create()
# 设置训练方法为弹性反向传播(RPROP)并更新权重
animals_net.setTrainMethod(cv2.ml.ANN_MLP_RPROP | cv2.ml.ANN_MLP_UPDATE_WEIGHTS)
# 设置激活函数为Sigmoid函数
animals_net.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
# 设置输入层，隐藏层和输出层的神经元个数（分别为3, 6, 4）
animals_net.setLayerSizes(np.array([3, 6, 4]))
# 设置停止条件
animals_net.setTermCriteria(( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ))

"""输入数组（动物的特征）
weight, length, teeth
"""

"""输出数组（分类）
dog（狗）, eagle（鹰）, dolphin（海豚） and dragon（龙）
"""
# 函数：随机生成狗的样本
def dog_sample():
    return [randint(10, 20), 1, randint(38, 42)]
# 函数：返回狗的分类(1的下标：0)
def dog_class():
    return [1, 0, 0, 0]

# 函数：随机生成秃鹰的样本
def condor_sample():
    return [randint(3,10), randint(3,5), 0]
# 函数：返回秃鹰的分类(1的下标：1)
def condor_class():
    return [0, 1, 0, 0]

# 函数：随机生成海豚样本
def dolphin_sample():
    return [randint(30, 190), randint(5, 15), randint(80, 100)]
# 函数：返回海豚的分类(1的下标：2)
def dolphin_class():
    return [0, 0, 1, 0]

# 函数：随机生成龙的样本
def dragon_sample():
    return [randint(1200, 1800), randint(30, 40), randint(160, 180)]
# 函数：返回龙的分类(1的下标：3)
def dragon_class():
    return [0, 0, 0, 1]

# 函数：记录样本及对应的分类
def record(sample, classification):
    return (np.array([sample], dtype=np.float32), np.array([classification], dtype=np.float32))

# 训练记录（样本+分类）列表
records = []
# 记录数（样本数）为5000
RECORDS = 5000
# 创建4类动物的数据，每类动物有5000个样本
for x in range(0, RECORDS):
    records.append(record(dog_sample(), dog_class()))
    records.append(record(condor_sample(), condor_class()))
    records.append(record(dolphin_sample(), dolphin_class()))
    records.append(record(dragon_sample(), dragon_class()))

# 训练次数
EPOCHS = 2
for e in range(0, EPOCHS):
    print "Epoch %d:" % e
    for sample, clas in records:
        animals_net.train(sample, cv2.ml.ROW_SAMPLE, clas)

## 测试训练结果
# 测试次数
TESTS = 100
# 狗的正确预测次数
dog_results = 0
for x in range(0, TESTS):
    # 预测随机生成的狗样本的分类
    clas = int(animals_net.predict(np.array([dog_sample()], dtype=np.float32))[0])
    # 输出预测结果
    print "class: %d" % clas
    # 若预测正确
    if (clas) == 0:
        # 正确预测次数增1
        dog_results += 1
    
# 秃鹰的正确预测次数
condor_results = 0
for x in range(0, TESTS):
    # 预测随机生成的秃鹰样本的分类
    clas = int(animals_net.predict(np.array([condor_sample()], dtype=np.float32))[0])
    # 输出预测结果
    print "class: %d" % clas
    # 增加正确预测次数
    if (clas) == 1:
        condor_results += 1

# 海豚的正确预测次数
dolphin_results = 0
for x in range(0, TESTS):
    # 预测随机生成的海豚样本的分类
    clas = int(animals_net.predict(np.array([dolphin_sample()], dtype=np.float32))[0])
    # 输出预测结果
    print "class: %d" % clas
    # 增加正确预测次数
    if (clas) == 2:
        dolphin_results += 1

# 龙的正确预测次数
dragon_results = 0
for x in range(0, TESTS):
    # 预测随机生成的龙样本的分类
    clas = int(animals_net.predict(np.array([dragon_sample()], dtype=np.float32))[0])
    # 输出预测结果
    print "class: %d" % clas
    # 增加正确预测次数
    if (clas) == 3:
        dragon_results += 1

# 输出预测的正确率
print "Dog accuracy: %f%%" % (dog_results / TESTS * 100)
print "condor accuracy: %f%%" % (condor_results / TESTS * 100)
print "dolphin accuracy: %f%%" % (dolphin_results / TESTS * 100)
print "dragon accuracy: %f%%" % (dragon_results / TESTS * 100)