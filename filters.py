# coding=utf-8
import cv2
import numpy
import utils

# 描边
def strokeEdges(src, dst, blurKsize = 7, edgeKsize = 5):
    if blurKsize >= 3: # 模糊核大小 > 3
        blurredSrc = cv2.medianBlur(src, blurKsize) # 中值模糊
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY) # 灰度化
    else:
        graySrc = cv2.cv2Color(src, cv2.COLOR_BGR2GRAY) # 直接灰度化
    
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize = edgeKsize) # 边缘化
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc) # 用于反相并归一化
    channels = cv2.split(src) # 分离通道
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha # 对每个通道的所有像素进行反相和归一化
    cv2.merge(channels, dst) # 合并处理后的所有通道

# 卷积过滤器(基类)
class VConvolutionFilter(object):
    """ A filter that applies a convolution to V (or all of BGR)."""

    def __init__ (self, kernel):
        self._kernel = kernel

    def apply (self, src, dst):
        """ Apply the filter with a BGR or gray source/destination."""
        cv2.filter2D(src, -1, self._kernel, dst)

# 锐化过滤器
class SharpenFilter(VConvolutionFilter):
    """ A sharpen filter with a 1-pixel radius."""
    
    def __init__ (self):
        kernel = numpy.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        VConvolutionFilter.__init__(self, kernel)

# 边缘检测过滤器
class FindEdgesFilter(VConvolutionFilter):
    """ An edge-finding filter with a 1-pixel radius."""

    def __init__ (self):
        kernel = numpy.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ])
        VConvolutionFilter.__init__(self, kernel)

# 模糊滤波器
class BlurFilter(VConvolutionFilter):
    """ An blur filter with a 2-pixel radius."""

    def __init__ (self):
        kernel = numpy.array([
            [0.04, 0.04, 0.04, 0.04, 0.04],
            [0.04, 0.04, 0.04, 0.04, 0.04],
            [0.04, 0.04, 0.04, 0.04, 0.04],
            [0.04, 0.04, 0.04, 0.04, 0.04],
            [0.04, 0.04, 0.04, 0.04, 0.04]
        ])
        VConvolutionFilter.__init__ (self, kernel)

# 浮雕滤波器
class EmbossFilter(VConvolutionFilter):
    """ An emboss filter with a 1-pixel raidus."""

    def __init__ (self):
        kernel = numpy.array([
            [-2, -1, 0],
            [-1,  1, 1],
            [ 0,  1, 2]
        ])
        VConvolutionFilter.__init__(self, kernel)