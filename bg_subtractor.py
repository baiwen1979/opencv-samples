# coding=utf-8
import cv2
# 打开系统默认摄像头
cap = cv2.VideoCapture('movies/traffic.flv')
# 使用MOG(Mixture Of Gaussians)2背景分割器
mog = cv2.createBackgroundSubtractorMOG2()
# 循环处理每一帧
while True:
    # 读取一帧
    ret, frame = cap.read()
    # 背景分割，返回前景掩码图像
    fgmask = mog.apply(frame)
    # 显示前景掩码图像
    cv2.imshow('frame', fgmask)
    # 按ESC退出
    if cv2.waitKey(30) & 0xff == 27:
        break

cap.release()
cv2.destroyAllWindows()