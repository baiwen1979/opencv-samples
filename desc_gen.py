# coding=utf-8
import os
import cv2
import numpy as np
from os import walk
from os.path import join
import sys

# 在指定的文件夹(folder)内的所有图片创建特征描述符(descripters)
def create_descriptors(folder):
    files = []
    for (dirpath, dirnames, filenames) in walk(folder):
        files.extend(filenames)
    for f in files:
        save_descriptor(folder, f, cv2.xfeatures2d.SIFT_create())
# 使用某种特征检测器(feature_detector)将指定路径图片(image_path)的特征描述符保存到指定的文件夹(folder)内
def save_descriptor(folder, image_path, feature_detector):    
    ignoreFileExts = ['.npy', '.DS_Store', '.pgm']
    if image_path[image_path.rindex('.'):] in ignoreFileExts:
        return
    # 加载图像
    print "reading %s" % image_path
    img = cv2.imread(join(folder, image_path), 0)
    # 使用指定的特征检测器进行特征检测和描述符的计算
    keypoints, descriptors = feature_detector.detectAndCompute(img, None)
    # 描述符文件
    descriptor_file = image_path.replace("png", "jpg")
    descriptor_file = descriptor_file.replace("jpg", "npy")
    # 存储描述符到文件中
    np.save(join(folder, descriptor_file), descriptors)

# 第一个命令行参数用作创建描述符的目录
if len(sys.argv) > 1:
    dir = sys.argv[1]
else:
    dir = 'images/anchors'

create_descriptors(dir)