# coding=utf-8
import cv2
import numpy as np

# 函数：blob特征检测
def detect_blob(img_src):
    # 初始化blob参数
    params = cv2.SimpleBlobDetector_Params()
    # Blob之间的最小距离
    # params.minDistBetweenBlobs = 0.0
    # 最小阈值
    # params.minThreshold = 90
    # 最大阈值
    # params.maxThreshold = 120
    # 根据惯性进行过滤
    params.filterByInertia = False
    params.minInertiaRatio = 0.0
    params.maxInertiaRatio = 0.6
    # 根据颜色进行过滤
    params.filterByColor = False
    # 根据面积过滤，设置最大与最小面积
    params.filterByArea = False
    params.minArea = 0.0
    params.maxArea = 2000.0
    # 根据圆度过滤，设置最大与最小圆度
    params.filterByCircularity = False
    params.minCircularity = 0.5
    params.maxCircularity = 1.0
    # 凸包形状分析 - 过滤凹包
    params.filterByConvexity = False
    params.minConvexity = 1.0
    
    # 创建blob检测器
    detector = cv2.SimpleBlobDetector_create(params)
    # 对源图像进行blob特征检测
    keypoints = detector.detect(img_src)
    # 绘制检测点
    img_kp = cv2.drawKeypoints(img_src, keypoints, np.array([]),
            (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # 返回包含检测点的图像
    return img_kp

# 函数：根据标准图像(img_t)检测图像(img_s)中的缺陷
def detect_defects(img_t, img_s):
    # 差分
    diff = cv2.absdiff(img_t, img_s)
    # 二值化
    ret, binary = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    # 形态(Morphology)化
    # (1) 获取3X3大小的椭圆形态结构化元素
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # (2) 膨胀
    dilated = cv2.dilate(diff, es, iterations = 3)
    # (3) 腐蚀
    eroded = cv2.erode(dilated, es, iterations = 2)
    # blob特征检测
    blob = detect_blob(eroded)
    return blob

def main():
    img_t = cv2.imread('images/varese2.jpg', cv2.IMREAD_GRAYSCALE)
    img_s = cv2.imread('images/varese5.jpg', cv2.IMREAD_GRAYSCALE)
    img_d = detect_defects(img_t, img_s)
    # 显示检测结果
    cv2.imshow("img_d", img_d)

    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()