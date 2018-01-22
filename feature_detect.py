# coding=utf-8
import cv2
import sys
import numpy as np

imgpath = sys.argv[1]
img = cv2.imread(imgpath)
alg = sys.argv[2]

# 选择特征检测算法
def fd(algorithm):
  algorithms = {
    "SIFT": cv2.xfeatures2d.SIFT_create(),
    "SURF": cv2.xfeatures2d.SURF_create(float(sys.argv[3]) if len(sys.argv) == 4 else 4000),
    "ORB": cv2.ORB_create()
  }
  return algorithms[algorithm]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 选择特征检测算法
fd_alg = fd(alg)
# 检测（关键点）并计算（描述符）
keypoints, descriptor = fd_alg.detectAndCompute(gray, None)
# 在原图上绘制关键点
img = cv2.drawKeypoints(image=img, outImage=img, keypoints = keypoints, flags = 4, color = (51, 163, 236))
# 输出关键点信息
print 'Number of key points: ', len(keypoints)
# 显示修改后的图像
cv2.imshow('keypoints', img)
while (True):
    if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
        break
cv2.destroyAllWindows()