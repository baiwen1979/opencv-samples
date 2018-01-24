# coding=utf-8
from os.path import join
from os import walk
import numpy as np
import cv2
from sys import argv

# 第一个命令行参数作为文件夹名称
folder = argv[1]
# 读取要查找的原目标图片
query = cv2.imread(join(folder, "tattoo_seed.jpg"), 0)

# 全局文件列表，图像列表和描述符列表
files = []
images = []
descriptor_files = []
for (dirpath, dirnames, filenames) in walk(folder):
    files.extend(filenames)
    for f in files:
        if f.endswith("npy") and f != "tattoo_seed.npy":
            descriptor_files.append(f)
    print descriptor_files

# 创建SIFT检测器
sift = cv2.xfeatures2d.SIFT_create()
# 检测并计算描述符
query_kp, query_ds = sift.detectAndCompute(query, None)

# 准备FLANN匹配器
FLANN_INDEX_KDTREE = 0
# 设置FLANN匹配器参数
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
# 创建FLANN匹配器
flann = cv2.FlannBasedMatcher(index_params, search_params)

# 指定最小匹配数
MIN_MATCH_COUNT = 10
# 用来记录犯罪嫌疑人及其匹配个数的字典
potential_culprits = {}

print ">> Initiating picture scan..."
for d in descriptor_files:
    print "--------- analyzing %s for matches ------------" % d
    # 加载描述符文件，对原目标文件的描述符执行FLANN的KNN匹配操作
    matches = flann.knnMatch(query_ds, np.load(join(folder, d)), k = 2)
    # 找出良好匹配（距离比 < 0.7)
    goodMatches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            goodMatches.append(m)
    # 若良好匹配的个数 > 指定的最小匹配个数，输出其描述符和匹配个数
    if len(goodMatches) > MIN_MATCH_COUNT:       
        print "%s is a match! (%d)" % (d, len(goodMatches))
    else: # 否则
        print "%s is not a match" % d
    # 记录当前犯罪嫌疑人的匹配个数
    potential_culprits[d] = len(goodMatches)
# 遍历犯罪嫌疑人表（字典）找出最大匹配数的犯罪嫌疑人
max_matches = None
potential_suspect = None
for culprit, matches in potential_culprits.iteritems():
    if max_matches == None or matches > max_matches:
        max_matches = matches
        potential_suspect = culprit
# 输出结果
print "potential suspect is %s" % potential_suspect.replace("npy", "").upper()