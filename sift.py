# coding=utf-8
import cv2
import sys
import numpy as np

# 打印所有关键点信息
def printKeyPoints(keypoints = None):
    if keypoints is None:
        return
    for keypoint in keypoints:
        printKeyPoint(keypoint)

# 打印单个关键点信息
def printKeyPoint(keypoint):
    print '{', \
          ' pt:', keypoint.pt,\
          ', size:', keypoint.size,\
          ', angle:', keypoint.angle,\
          ', response:', keypoint.response,\
          ', octave:', keypoint.octave,\
          ', class_id:', keypoint.class_id,\
          '}'
    
# 主程序
if __name__ == '__main__':
    # 通过命令行参数获得图像路径
    imgpath = sys.argv[1]
    # 加载图像
    img = cv2.imread(imgpath)
    # 转化色彩空间为灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 创建SIFT对象
    sift = cv2.xfeatures2d.SIFT_create()
    # 通过SIFT对象对灰度图像进行关键点检测，并对关键点周围的区域计算特征向量
    # 检测和计算完成后，返回关键点信息和描述符
    keypoints, descriptor = sift.detectAndCompute(gray, None)
    
    # 在图像上绘制关键点
    # 标志：flags = 4 (cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINT)表示对每个关键点绘制圆圈和方向
    img = cv2.drawKeypoints(image = img, outImage= img, keypoints = keypoints, 
        flags = 4, color = (51, 163, 236))
    
    # 输出关键点信息
    print 'Number of key points: ', len(keypoints)
    # printKeyPoints(keypoints)

    # 显示修改后的图像
    cv2.imshow('sift_keypoints', img)
    while (True):
        if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
            break
    cv2.destroyAllWindows()