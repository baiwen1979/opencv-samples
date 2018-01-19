# coding=utf-8
import numpy as np
import cv2

def update(val = 0):
    # 设置匹配块的大小
    stereo.setBlockSize(cv2.getTrackbarPos('window_size', 'disparity'))
    # 设置单值性比率
    stereo.setUniquenessRatio(cv2.getTrackbarPos('uniquenessRatio', 'disparity'))
    # 设置斑点窗口的大小，以进行斑点过滤
    stereo.setSpeckleWindowSize(cv2.getTrackbarPos('speckleWindowSize', 'disparity'))
    # 设置最大已连接部分的最大视差变化
    stereo.setSpeckleRange(cv2.getTrackbarPos('speckleRange', 'disparity'))
    # 设置左右视差检查中最大允许的偏差
    stereo.setDisp12MaxDiff(cv2.getTrackbarPos('disp12MaxDiff', 'disparity'))

    print '正在计算视差...'
    # 计算视差图
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    cv2.imshow('left', imgL)
    cv2.imshow('disparity', (disp - min_disp) / num_disp)

if __name__ == "__main__":
    # 初始参数
    window_size = 5
    min_disp = 16
    num_disp = 192 - min_disp
    blockSize = window_size
    uniquenessRatio = 1
    speckleRange = 3
    speckleWindowSize = 3
    disp12MaxDiff = 200
    p1 = 600
    p2 = 2400
    # 加载左右眼图像
    imgL = cv2.imread('images/color1_small.jpg')
    imgR = cv2.imread('images/color2_small.jpg')
    # 创建命名窗口
    cv2.namedWindow('disparity')
    # 创建用来调整参数的滑动条控件
    cv2.createTrackbar('speckleRange', 'disparity', speckleRange, 50, update)
    cv2.createTrackbar('window_size', 'disparity', window_size, 21, update)
    cv2.createTrackbar('speckleWindowSize', 'disparity', speckleWindowSize, 200, update)
    cv2.createTrackbar('uniquenessRatio', 'disparity', uniquenessRatio, 50, update)
    cv2.createTrackbar('disp12MaxDiff', 'disparity', disp12MaxDiff, 250, update)
    # 根据初始参数创建StereoSGBM实例，用于计算视差图
    stereo = cv2.StereoSGBM_create(
        minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = window_size,
        uniquenessRatio = uniquenessRatio,
        speckleRange = speckleRange,
        speckleWindowSize = speckleWindowSize,
        disp12MaxDiff = disp12MaxDiff,
        P1 = p1,
        P2 = p2
    )

    update()
    cv2.waitKey()
    cv2.destroyAllWindows()
