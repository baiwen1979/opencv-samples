# coding=utf-8
import cv2, numpy as np

# 实际测量的鼠标移动轨迹
measurements = []
# 预测鼠标移动轨迹
predictions = []
# 创建大小为800X800的空帧
frame = np.zeros((800, 800, 3), np.uint8)
# 初始化测量坐标数组
last_measurement = current_measurement = np.array((2, 1), np.float32) 
# 初始化预测坐标数组
last_prediction = current_prediction = np.zeros((2, 1), np.float32)

# 鼠标移动的回调函数，用来绘制跟踪结果
def mousemove(event, x, y, s, p):
    # 使用之前定义的全局变量
    global frame, current_measurement, measurements, last_measurement, current_prediction, last_prediction
    # 存储当前测试和预测为上一次测量和预测
    last_prediction = current_prediction
    last_measurement = current_measurement
    # 获取当前测量的鼠标位置
    current_measurement = np.array([[np.float32(x)], [np.float32(y)]])
    # 用当前测量来校正卡尔曼滤波器
    kalman.correct(current_measurement)
    # 计算当前卡尔曼预测值
    current_prediction = kalman.predict()

    # 上次测量位置
    lmx, lmy = last_measurement[0], last_measurement[1]
    # 当前测量位置
    cmx, cmy = current_measurement[0], current_measurement[1]
    # 上次预测位置
    lpx, lpy = last_prediction[0], last_prediction[1]
    # 当前预测位置
    cpx, cpy = current_prediction[0], current_prediction[1]
    # 绘制从上次测量位置到当前测量位置的线段(绿色)
    cv2.line(frame, (lmx, lmy), (cmx, cmy), (0, 100, 0))
    # 绘制从上次预测位置到当前预测位置的线段(红色)
    cv2.line(frame, (lpx, lpy), (cpx, cpy), (0, 0, 200))

# 创建命名窗口
cv2.namedWindow("kalman_tracker")
# 为命名窗口注册鼠标移动事件处理回调函数
cv2.setMouseCallback("kalman_tracker", mousemove);

# 创建卡尔曼滤波器
# 参数：状态维度(dynamParams):4
#      测量维度(measureParams):2
#      控制维度(controlParams):1
kalman = cv2.KalmanFilter(4, 2, 1)
# 设置测量矩阵
kalman.measurementMatrix = np.array([[1, 0, 0, 0],[0, 1, 0, 0]], np.float32)
# 设置变换矩阵
kalman.transitionMatrix = np.array([[1, 0, 1, 0],[0, 1, 0, 1],[0, 0, 1, 0],[0, 0, 0, 1]], np.float32)
# 噪声处理覆盖矩阵
kalman.processNoiseCov = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]], np.float32) * 0.03

while True:
    cv2.imshow("kalman_tracker", frame)
    # 按ESC退出
    if (cv2.waitKey(30) & 0xFF) == 27:
        break
    # 按Q键保存结果并退出
    if (cv2.waitKey(30) & 0xFF) == ord('q'):
        cv2.imwrite('images/kalman.jpg', frame)
        break
# 关闭所有窗口
cv2.destroyAllWindows()