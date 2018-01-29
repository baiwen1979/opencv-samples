# coding=utf-8

import cv2
import numpy as np
# 创建ANN多层感知器MLP(Multilayer Perceptron)
ann = cv2.ml.ANN_MLP_create()
# 设置网络拓扑结构：输入层的大小为9，隐藏层的大小为5，输出层的大小为9
ann.setLayerSizes(np.array([9, 5, 9], dtype=np.uint8))
# 设置训练方法为反向传播(BACKPROP),根据分类误差调整权重
ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
# 进行有监督学习（训练）
## 样本数据
samples = np.array([[1.2, 1.3, 1.9, 2.2, 2.3, 2.9, 3.0, 3.2, 3.3]], dtype=np.float32)
## 样本布局
layout = cv2.ml.ROW_SAMPLE
## 对应的分类
responses = np.array([[0,0,0,0,0,1,0,0,0]], dtype=np.float32)
# 训练
ann.train(samples, layout, responses)
# 预测
print ann.predict(np.array([[1.4, 1.5, 1.2, 2., 2.5, 2.8, 3., 3.1, 3.8]], dtype=np.float32))