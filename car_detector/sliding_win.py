# coding=utf-8

# 生成器函数：返回给定的图像中（从左向右）滑动窗口区域中的图像
def sliding_window(image, step, window_size):
    '''
    返回给定的图像中（从左向右）滑动窗口区域中的图像.
    [参数]image 图像,
    [参数]step 滑动步长,
    [参数]window_size 窗口大小.
    [返回]窗口区域中的图像
    '''
    for y in xrange(0, image.shape[0], step):
        for x in xrange(0, image.shape[1], step):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])