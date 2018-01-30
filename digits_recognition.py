# coding=utf-8

import cv2
import numpy as np
import digits_ann as ANN

# 函数：判断一个矩形(r1)是否完全包含在另一个矩形(r2)中
def inside(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if (x1 > x2) and (y1 > y2) and (x1 + w1 < x2 + w2) and (y1 + h1 < y2 + h2):
        return True
    else:
        return False

# 函数：将数字周围的矩形转化为包围数字的正方形
def wrap_digit(rect):
    x, y, w, h = rect
    padding = 5
    hcenter = x + w / 2
    vcenter = y + h / 2
    roi = None
    if (h > w):
        w = h
        x = hcenter - (w / 2)
    else:
        h = w
        y = vcenter - (h / 2)
    return (x - padding, y - padding, w + padding, h + padding)

# 创建58个隐藏层的ANN，使用50000个样本进行5个周期的训练
ann, test_data = ANN.train(ANN.create_ANN(58), 50000, 5)
# 用来显示预测结果的字体
font = cv2.FONT_HERSHEY_SIMPLEX

# 加载包含数字的图片
path = "./images/MNISTsamples.png"
# path = "images/numbers.jpg"
img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
# 转换为灰度图
bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 使用高斯模糊进行平滑处理
bw = cv2.GaussianBlur(bw, (7, 7), 0)

# 使用阈值和形态学操作方法来确保数字能从背景中正确分离，以增加预测的成功率
## 逆二值化（黑底白字）
ret, thbw = cv2.threshold(bw, 127, 255, cv2.THRESH_BINARY_INV)
## 腐蚀
thbw = cv2.erode(thbw, np.ones((2, 2), np.uint8), iterations = 2)

# 识别图像中的轮廓
image, cntrs, hier = cv2.findContours(thbw.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 轮廓矩形列表
rectangles = []

# 对于每个轮廓
for c in cntrs:
    # 计算轮廓的矩形边框      
    r = x, y, w, h = cv2.boundingRect(c)
    # 计算轮廓面积
    a = cv2.contourArea(c)
    # 计算整个图像的面积
    b = (img.shape[0] - 3) * (img.shape[1] - 3)
    
    # 放弃所有完全包含在其它矩形中的矩形
    is_inside = False
    for q in rectangles:
        if inside(r, q):
            is_inside = True
            break
    if not is_inside:
        if not a == b:
          rectangles.append(r)
# 对所有好的数字轮廓矩形：
for r in rectangles:
    # 计算以要识别数字为中心的包围正方形
    x, y, w, h = wrap_digit(r)
    # 再图像上绘制数字轮廓矩形
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 二值化图像中提取上述正方形感兴趣区域ROI
    roi = thbw[y: y + h, x: x + w]
    
    try:
        # 预测ROI中数字图像的分类（识别出的数字）
        digit_class = int(ANN.predict(ann, roi.copy())[0])
    except:
        continue
    # 在原图中显示识别（预测）结果
    cv2.putText(img, "%d" % digit_class, (x, y - 1), font, 1, (0, 255, 0))
# 显示二值化图像
cv2.imshow("thbw", thbw)
# 显示包含数字轮廓和识别结果的图像
cv2.imshow("contours", img)
# 存储到文件
cv2.imwrite("images/sample.jpg", img)
cv2.waitKey()