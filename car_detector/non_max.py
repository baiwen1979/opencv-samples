# coding=utf-8
# 导入需要的模块
import numpy as np

# 函数：非最大抑制函数
def non_max_suppression_fast(boxes, overlapThresh):
    '''
    非最大抑制函数：对指定的多个矩形框(boxes)进行评分并排序，从评分最高的矩形开始，
    消除所有重叠超过一定阈值(overlapThresh)的矩形.
    消除的规则是计算相交的区域，并看这些相交的区域是否大于某一阈值
    '''
    # 如果没有矩形框，则返回控列表
    if len(boxes) == 0:
        return []

    # 如果矩形框的边界为整数，则将其转化为浮点数，因为要做除法运算
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # 创建所挑选矩形框索引的列表
    pick = []

    # 获得矩形框边界点的坐标
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # 及其评分
    scores = boxes[:,4]
    # 计算边界矩形框的面积并根据评分进行排序
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)[::-1]

    # 从索引列表中挑选重叠区域不小于指定阈值的矩形框索引
    while len(idxs) > 0:
        # 获取索引列表中最后的索引值，并将其添加到已挑选索引列表中
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # 查找边界矩形框的最大(x,y)起点坐标（左上角）和最小终点坐标（右下角）
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # 计算边界矩形框的宽度和高度
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # 计算重叠率
        overlap = (w * h) / area[idxs[:last]]
        # 删除所有重叠率大于指定阈值的边界矩形框索引
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # 使用整数类型返回挑选的矩形框列表
    return boxes[pick].astype("int")