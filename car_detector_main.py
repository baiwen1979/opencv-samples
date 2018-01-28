# coding=utf-8
import cv2
import numpy as np
from car_detector.detector import car_detector, bow_features
from car_detector.pyramid import pyramid
from car_detector.non_max import non_max_suppression_fast as nms
from car_detector.sliding_win import sliding_window
import urllib

# 函数：判断两个数是否足够接近
def in_range(number, test, thresh = 0.2):
    return abs(number - test) < thresh

# 测试图像的路径
test_image = "images/cars.jpg"
img_path = "images/test.jpg"

urllib.urlretrieve(test_image, img_path)
# 获得训练好的SVM模型和特征提取器
svm, extractor = car_detector()
# 创建SIFT实例
detect = cv2.xfeatures2d.SIFT_create()
# 滑动窗口的宽度和高度
w, h = 100, 40
img = cv2.imread(img_path)
# 矩形列表
rectangles = []
# 计数器
counter = 1
# 缩放因子
scaleFactor = 1.25
# 缩放比例
scale = 1
# 显示字体
font = cv2.FONT_HERSHEY_PLAIN

# 遍历图像金字塔中的每个图像
for resized in pyramid(img, scaleFactor): 
    # 计算图像金字塔中当前图像大小相对于原图的缩放比例 
    scale = float(img.shape[1]) / float(resized.shape[1])
    # 对于当前金字塔图像中每个滑动窗口对应的感兴趣区域（ROI）
    for (x, y, roi) in sliding_window(resized, 20, (100, 40)):
        # 忽略不是100X40的感兴趣区域
        if roi.shape[1] != w or roi.shape[0] != h:
            continue

        try:
            # 提取当前ROI的BOW特征
            bf = bow_features(roi, extractor, detect)
            # 使用支持向量机当前ROI中的图像分类进行预测
            _, result = svm.predict(bf)
            # 使用可选参数flags进行预测，以返回预测评分
            a, res = svm.predict(bf, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT | cv2.ml.STAT_MODEL_UPDATE_MODEL)
            # 输出当前ROI预测分类，评分和预测结果
            print "Class: %d, Score: %f, a: %s" % (result[0][0], res[0][0], res)
            # 预测评分
            score = res[0][0]
            # 如果预测分类为正例（1）
            if result[0][0] == 1:
                # 评分越低，置信度越高，表示分类越正确
                if score < -1.0: # 好的分类
                    rx, ry, rx2, ry2 = int(x * scale), int(y * scale), int((x + w) * scale), int((y + h) * scale)
                    # 将当前滑动窗口的计算坐标（即用图像金字塔当前层数的尺度乘以当前坐标）
                    # 添加了矩形列表中
                    rectangles.append([rx, ry, rx2, ry2, abs(score)])
        except:
            pass
        counter += 1 

# 非最大化抑制
windows = np.array(rectangles)
boxes = nms(windows, 0.25)

# 显示并打印结果
for (x, y, x2, y2, score) in boxes:
    print x, y, x2, y2, score
    cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 1)
    cv2.putText(img, "%f" % score, (int(x), int(y)), font, 1, (0, 255, 0))

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()