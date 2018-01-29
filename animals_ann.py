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
# 函数：返回狗的分类(1)
def dog_class():
    return [1, 0, 0, 0]

# 函数：随机生成秃鹰的样本
def condor_sample():
    return [randint(3,10), randint(3,5), 0]
# 函数：返回秃鹰的分类(2)
def condor_class():
    return [0, 1, 0, 0]

# 函数：随机生成海豚样本
def dolphin_sample():
    return [randint(30, 190), randint(5, 15), randint(80, 100)]
# 函数：返回海豚的分类(3)
def dolphin_class():
    return [0, 0, 1, 0]

# 函数：随机生成龙的样本
def dragon_sample():
    return [randint(1200, 1800), randint(30, 40), randint(160, 180)]
# 函数：返回龙的分类(4)
def dragon_class():
    return [0, 0, 0, 1]

# 函数：记录样本及对应的分类
def record(sample, classification):
    return (np.array([sample], dtype=np.float32), np.array([classification], dtype=np.float32))

records = []

"""
SAMPLES = 5000
for x in range(0, SAMPLES):
  print "Samples %d/%d" % (x, SAMPLES)
  animals_net.train(np.array([dog_sample()], dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array([dog_class()], dtype=np.float32))
  animals_net.train(np.array([condor_sample()], dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array([condor_class()], dtype=np.float32))
  animals_net.train(np.array([dolphin_sample()], dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array([dolphin_class()], dtype=np.float32))
  animals_net.train(np.array([dragon_sample()], dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array([dragon_class()], dtype=np.float32))
"""
# 记录数（样本数）为5000
RECORDS = 5000
for x in range(0, RECORDS):
    records.append(record(dog_sample(), dog_class()))
    records.append(record(condor_sample(), condor_class()))
    records.append(record(dolphin_sample(), dolphin_class()))
    records.append(record(dragon_sample(), dragon_class()))

EPOCHS = 2
for e in range(0, EPOCHS):
    print "Epoch %d:" % e
    for t, c in records:
        animals_net.train(t, cv2.ml.ROW_SAMPLE, c)


TESTS = 100
dog_results = 0
for x in range(0, TESTS):
    clas = int(animals_net.predict(np.array([dog_sample()], dtype=np.float32))[0])
    print "class: %d" % clas
    if (clas) == 0:
        dog_results += 1

condor_results = 0
for x in range(0, TESTS):
    clas = int(animals_net.predict(np.array([condor_sample()], dtype=np.float32))[0])
    print "class: %d" % clas
    if (clas) == 1:
        condor_results += 1

dolphin_results = 0
for x in range(0, TESTS):
    clas = int(animals_net.predict(np.array([dolphin_sample()], dtype=np.float32))[0])
    print "class: %d" % clas
    if (clas) == 2:
        dolphin_results += 1

dragon_results = 0
for x in range(0, TESTS):
    clas = int(animals_net.predict(np.array([dragon_sample()], dtype=np.float32))[0])
    print "class: %d" % clas
    if (clas) == 3:
        dragon_results += 1

print "Dog accuracy: %f%%" % (dog_results)
print "condor accuracy: %f%%" % (condor_results)
print "dolphin accuracy: %f%%" % (dolphin_results)
print "dragon accuracy: %f%%" % (dragon_results)