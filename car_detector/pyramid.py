# coding=utf-8
import cv2

# 函数：根据指定的缩放因子调整图像的大小
def resize(img, scaleFactor):
    '''
    根据指定的缩放因子(scaleFactor)调整图像(img)的大小
    '''
    w = int(img.shape[1] * (1 / scaleFactor))
    h = int(img.shape[0] * (1 / scaleFactor))
    return cv2.resize(img, (w, h), interpolation = cv2.INTER_AREA)

# 生成器函数：建立图像金字塔
def pyramid(image, scale = 1.5, minSize = (200, 80)):
    '''
    按照指定的缩放比例(scale)和最小宽高约束(minSize),建立图像金字塔
    返回图像金字塔中的各个大小不同（按比例缩放）的图像
    '''
    yield image

    while True:
        image = resize(image, scale)
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        yield image