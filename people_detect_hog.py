# coding=utf-8
import cv2
import numpy as np

# 判断矩形i是否在矩形o中
def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih

# 框住检测到的人
def draw_person(image, person):
    x, y, w, h = person
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

# 加载图像
img = cv2.imread("images/people.jpg")
# 创建HOG描述符对象
hog = cv2.HOGDescriptor()
# 设置SVM检测器为HOG描述符的默认人物检测器
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# 进行多尺度人物检测：窗口大小为8X8，缩放因子为1.05
found, w = hog.detectMultiScale(img, winStride = (8, 8), scale = 1.05)
# 用于进行检测结果过滤的列表
found_filtered = []
# 遍历检测结果，过滤掉不含有检测目标的区域
for ri, r in enumerate(found):
    for qi, q in enumerate(found):
        if ri != qi and is_inside(r, q):
            break
    else:
        found_filtered.append(r)
# 框住所有检测目标
for person in found_filtered:
    draw_person(img, person)
# 显示结果
cv2.imshow("people detection", img)  
cv2.waitKey(0)
cv2.destroyAllWindows()