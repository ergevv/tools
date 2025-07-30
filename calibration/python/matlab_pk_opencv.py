# -*- coding: utf-8 -*-
"""

@author: erge
"""
import os
import numpy as np
import cv2
import glob
import yaml
import re
import scipy.io


# 加载 .mat 文件
mat_data = scipy.io.loadmat('/home/erge/work/slam/kalibr_workspace/mydata/kalibr.mat')

# 提取数据
PatternPoints = mat_data['PatternPoints']
WorldPoints = mat_data['WorldPoints']
# 加载 YAML 配置文件
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config
# 自定义排序规则
def custom_sort_key(s):
    # 使用正则表达式提取文件名中的数字部分
    match = re.search(r"(\d+)\.png$", s)  # 匹配以数字结尾并以 .png 结尾的部分
    if match:
        return int(match.group(1))  # 返回数字部分作为排序键
    return float('inf')  # 如果没有匹配到数字，放到最后



print(PatternPoints.shape)  # 输出 (190, 2, 51)

config = load_config('calibration/config.yaml')

# 解析配置参数
pattern_size = tuple(config['calibration']['pattern_size'])  # 标定板圆点的行列数
square_size = config['calibration']['square_size']           # 圆点间距
left_single_image = config['calibration']['left_single_image']             # 标定图像路径
righe_single_image = config['calibration']['righe_single_image']             # 标定图像路径
left_image = config['calibration']['left_image']             # 标定图像路径
right_image = config['calibration']['right_image']             # 标定图像路径
show_image = config['calibration']['show_image']
save_results = config['output']['save_results']          # 是否保存结果
result_file = config['output']['result_file']                # 结果文件名



# monocular camera calibration
images_l = sorted(glob.glob(left_single_image),key=custom_sort_key)
images_r = sorted(glob.glob(righe_single_image),key=custom_sort_key)

objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = WorldPoints


# 存储对象点和图像点的数组
objpoints = []  # 3D 点在世界坐标系中的位置
imgpoints = []  # 2D 点在图像平面中的位置
i=0
for fname in images_l:
    objpoints.append(objp.astype(np.float32))
    tem = PatternPoints[:,:,i]
    tem = tem.reshape(190, 1, 2).astype(np.float32)
    imgpoints.append(tem)
    i+=1
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("相机内参矩阵:\n", mtx)
print("畸变系数:\n", dist)
print('重投影误差:\n', ret) #ret 是 OpenCV 默认返回的总重投影误差，基于均方根误差计算。
all_error =[]
mean_error = 0
for i in range(len(objpoints)):
    imgpoints_proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints_proj, cv2.NORM_L2) / len(imgpoints_proj)
    all_error.append(error)
    mean_error += error
print("Total Mean Reprojection Error: ", mean_error / len(objpoints))