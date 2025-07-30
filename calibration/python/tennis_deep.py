# -*- coding: utf-8 -*-
"""
@author: erge
2025/04/12
"""
import os
import numpy as np
import cv2
import glob
import yaml
import re
import scipy.io
import json
import math


# 加载 YAML 配置文件
def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


# 加载 .mat 文件
base_dir = "calibration/matlab"
date = "20250604.mat"
mat_path = os.path.join(base_dir, date)
mat_data = scipy.io.loadmat(mat_path)

# 提取数据
worldPoints = mat_data["worldPoints"]
leftSingleImagePoints = mat_data["leftSingleImagePoints"]
rightSingleImagePoints = mat_data["rightSingleImagePoints"]
leftStereoImagePoints = mat_data["leftStereoImagePoints"]
rightStereoImagePoints = mat_data["rightStereoImagePoints"]
imageSize = mat_data["imageSize"]
imageSize = [imageSize[0, 1], imageSize[0, 0]]

# 读取对应图像
imageNames = mat_data['image_name_stereo'][0][0]
image_path = "/mnt/disk2/ubuntu/calibration/20250604"
path_left = ["leftImage","left","",".png"]
path_right = ["rightImage","right","",".png"]

for image_name in imageNames:
    image_name_left = os.path.join(image_path , path_left[0], path_left[1] + str(image_name) + path_left[3])
    image_name_right = os.path.join(image_path , path_right[0], path_right[1] + str(image_name) + path_right[3])
    image_left = cv2.imread(image_name_left)
    image_right = cv2.imread(image_name_right)






# 解析配置参数
config = load_config("calibration/config.yaml")
pattern_size = tuple(config["calibration"]["pattern_size"])
save_results = config["output"]["save_results"]  # 是否保存结果
result_file = config["output"]["result_file"]  # 结果文件名
fix_translation = np.array(config["calibration"]["fix_translation"], dtype=np.float64)
square_size = config["calibration"]["square_size"]  # 圆点间距
single_mode = config["calibration"]["single_mode"]

if single_mode == 1:
    single_flags = None
elif single_mode == 2:
    single_flags = cv2.CALIB_FIX_K3
elif single_mode == 3:
    single_flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
elif single_mode == 4:
    single_flags = cv2.CALIB_ZERO_TANGENT_DIST
elif single_mode == 5:
    single_flags = (
        cv2.CALIB_FIX_ASPECT_RATIO
        + cv2.CALIB_ZERO_TANGENT_DIST
        + cv2.CALIB_SAME_FOCAL_LENGTH
        + cv2.CALIB_RATIONAL_MODEL
        + cv2.CALIB_FIX_K3
        + cv2.CALIB_FIX_K4
        + cv2.CALIB_FIX_K5
    )
else:
    raise ValueError("Invalid mode")

stereo_mode = config["calibration"]["stereo_mode"]
if stereo_mode == 1:
    stereo_flags = cv2.CALIB_FIX_INTRINSIC
    fix_translation = None
    init_R = None
elif stereo_mode == 2:
    stereo_flags = (
        cv2.CALIB_FIX_INTRINSIC
    )  # | cv2.CALIB_USE_EXTRINSIC_GUESS  # 固定平移向量
    init_R = np.eye(3)
elif stereo_mode == 3:
    stereo_flags = None
    fix_translation = None
    init_R = None
else:
    raise ValueError("Invalid mode")


# 定义终止条件
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10000, 1e-6)


# 世界点
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = worldPoints


# 左相机标定
leftSingleObj = []  # 3D 点在世界坐标系中的位置
leftSinglePoints = []  # 2D 点在图像平面中的位置

for i in range(leftSingleImagePoints.shape[-1]):
    leftSingleObj.append(objp)
    tem = leftSingleImagePoints[:, :, i]
    tem = tem.reshape(190, 1, 2).astype(np.float32)
    leftSinglePoints.append(tem)
# 设置标志位：禁用切向畸变


left_ret, left_mtx, left_dist, left_rvecs, left_tvecs = cv2.calibrateCamera(
    leftSingleObj,
    leftSinglePoints,
    imageSize,
    None,
    None,
    flags=single_flags,
    criteria=criteria,
)

# left_ret, left_mtx, left_dist, left_rvecs, left_tvecs = cv2.calibrateCamera(leftSingleObj, leftSinglePoints, imageSize, None, None)

print("左相机内参矩阵:\n", left_mtx)
print("左相机畸变系数:\n", left_dist)
print("左相机重投影误差:\n", left_ret)

# 右相机标定
rightSingleObj = []  # 3D 点在世界坐标系中的位置
rightSinglePoints = []  # 2D 点在图像平面中的位置

for i in range(rightSingleImagePoints.shape[-1]):
    rightSingleObj.append(objp)
    tem = rightSingleImagePoints[:, :, i]
    tem = tem.reshape(190, 1, 2).astype(np.float32)
    rightSinglePoints.append(tem)

right_ret, right_mtx, right_dist, right_rvecs, right_tvecs = cv2.calibrateCamera(
    rightSingleObj,
    rightSinglePoints,
    imageSize,
    None,
    None,
    flags=single_flags,
    criteria=criteria,
)

print("右相机内参矩阵:\n", right_mtx)
print("右相机畸变系数:\n", right_dist)
print("右相机重投影误差:\n", right_ret)


# 双目标定
stereoObj = []
leftStereoPoints = []
rightStereoPoints = []
for i in range(leftStereoImagePoints.shape[-1]):
    stereoObj.append(objp)
    tem = leftStereoImagePoints[:, :, i]
    tem = tem.reshape(190, 1, 2).astype(np.float32)
    leftStereoPoints.append(tem)

    tem = rightStereoImagePoints[:, :, i]
    tem = tem.reshape(190, 1, 2).astype(np.float32)
    rightStereoPoints.append(tem)


ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
    stereoObj,
    leftStereoPoints,
    rightStereoPoints,
    left_mtx,
    left_dist,
    right_mtx,
    right_dist,
    imageSize,
    R=init_R,
    T=fix_translation,
    flags=stereo_flags,
    criteria=criteria,
)
print("左立体内参矩阵:\n", mtx_l)
print("左立体畸变系数:\n", dist_l)

print("右立体内参矩阵:\n", mtx_r)
print("右立体畸变系数:\n", dist_r)
print("立体重投影误差:\n", ret)
print("外参旋转矩阵:\n", R)
print("外参平移矩阵:\n", T)

norm = np.linalg.norm(T)
print("外参平移向量的模:", norm)  # 输出: 向量的模: 5.0

if save_results:
    # 创建一个字典来存储所有的矩阵
    data = {
        "左立体内参矩阵": mtx_l.tolist(),
        "左立体畸变系数": dist_l.tolist(),
        "右立体内参矩阵": mtx_r.tolist(),
        "右立体畸变系数": dist_r.tolist(),
        "外参旋转矩阵": R.tolist(),
        "外参平移矩阵": T.tolist(),
        "立体重投影误差": ret,
        "外参平移向量的模": norm,
    }

    # 将字典转换为 JSON 字符串，并写入到 txt 文件
    mode = os.path.join(result_file, str(single_mode) + str(stereo_mode) + ".json")

    # 将字典转换为 JSON 字符串
    with open(mode, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print("标定数据已保存为 JSON")


# 左相机内参矩阵：
#    1.0e+03 *

#     7.2082         0         0
#          0    7.2045         0
#     1.3023    1.0015    0.0010

#    1.0e+03 *

#     7.2082         0         0
#          0    7.2045         0
#     1.3023    1.0015    0.0010

# 右相机内参矩阵：
#    1.0e+03 *

#     7.2364         0         0
#          0    7.2376         0
#     1.3114    0.9691    0.0010

#    1.0e+03 *

#     7.2364         0         0
#          0    7.2376         0
#     1.3114    0.9691    0.0010

# 径向畸变系数（左相机）：
#     0.0745   -0.7222

#     0.0745   -0.7222

# 径向畸变系数（右相机）：
#     0.1097   -0.8200

#     0.1097   -0.8200

# 旋转矩阵R（从左相机到右相机）：
#     0.9983   -0.0188   -0.0549
#     0.0147    0.9971   -0.0747
#     0.0562    0.0737    0.9957

# 平移向量t（从左相机到右相机）：
#     1.2047 -457.3166  150.9995

#   481.6023

# cam_left = R * cam_right + t


# 读取对应双目图像的左相机图像，计算其标定板平面方程
# 理论上双目根据求计算的3D点与平面方程的距离小于半个网球（63.5/2），且球在平面前方

radius = 63.5 / 2

# 求解PnP，计算单目距离
for i in range(leftStereoImagePoints.shape[-1]):
    leftSingleObj.append(objp)
    centers = leftStereoImagePoints[:, :, i]
    centers = centers.reshape(190, 1, 2).astype(np.float32)

    ret, rvec, tvec = cv2.solvePnP(objp, centers, left_mtx, left_dist)
    R, _ = cv2.Rodrigues(rvec)

    reprojected_points, _ = cv2.projectPoints(objp, rvec, tvec, left_mtx, left_dist)
    error = cv2.norm(centers, reprojected_points, cv2.NORM_L2) / len(centers)

    # 转换到相机坐标系
    cam_points = (R @ objp.T + tvec).T
    # 平面拟合
    centroid = np.mean(cam_points, axis=0)
    A = cam_points - centroid
    U, S, Vt = np.linalg.svd(A)
    n = Vt[2, :]  # 法向量 [a, b, c]
    n /= np.linalg.norm(n)  # 单位化
    d = -np.dot(n, centroid)  # 平面方程参数 d

    # 判断法向量方向，保证法向量与相机同向
    if d < 0:
        n = -n
        d = -d

    d = d - radius * math.sqrt(
        n[0] ** 2 + n[1] ** 2 + n[2] ** 2
    )  # 沿着法向量靠近球心，去除球心影响

    # 平面方程: n[0]*X + n[1]*Y + n[2]*Z + d = 0
    plane_equation = f"{n[0]:.6f} * X + {n[1]:.6f} * Y + {n[2]:.6f} * Z + {d:.6f} = 0"
    depth = -d/n[2]
    print("平面方程（相机坐标系）:", plane_equation,"  球心距离:", depth)



# 计算

