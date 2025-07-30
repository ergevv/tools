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

# 获取当前 OpenCV 使用的线程数
# print("Current number of threads:", cv2.getNumThreads())
# # 设置 OpenCV 使用的线程数为 1（禁用多线程）
# cv2.setNumThreads(1)
# # 验证设置是否生效
# print("Updated number of threads:", cv2.getNumThreads())
pid = os.getpid()
print(f"Current process ID: {pid}")
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


if __name__ == "__main__":
    # 加载配置文件
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

    if not os.path.exists(result_file):
        os.makedirs(result_file)

    # 准备对象点，真实的世界坐标系中的点
    objp = np.zeros((pattern_size[0] * pattern_size[1],1, 3), np.float32)
    objp[:, :,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 1,2)
    objp *= square_size

    # 存储对象点和图像点的数组
    objpoints = []  # 3D 点在世界坐标系中的位置
    imgpoints = []  # 2D 点在图像平面中的位置

    # 单目标定：左相机
    # 创建一个可调整大小的窗口
    cv2.namedWindow('Detected Circles', cv2.WINDOW_NORMAL)
    screen_width = 1920
    screen_height = 1200
    cv2.resizeWindow('Detected Circles', screen_width, screen_height)
    for fname in images_l:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Detected Circles', gray)
        # cv2.waitKey(0)
        # 寻找非对称圆标定板
        params = cv2.SimpleBlobDetector_Params()
        # 调整参数
        params.minArea = 25      # 最小面积
        params.maxArea = 5000      # 最大面积
        params.minCircularity = 0.8  # 最小圆度
        params.minConvexity = 0.95  # 最小凸性
        params.minInertiaRatio = 0.1  # 最小惯性比
        params.filterByColor = True  # 启用颜色过滤
        # params.blobColor = 255      # 检测白色斑点
        params.minThreshold = 50    # 最小阈值
        params.maxThreshold = 220   # 最大阈值
        params.thresholdStep = 10   # 阈值步长
        detector = cv2.SimpleBlobDetector_create(params)

        # 创建 CirclesGridFinderParameters 并手动设置参数
        # parameters = cv2.CirclesGridFinderParameters()
        # parameters.minDensity = 10          # 减小最小密度阈值
        # parameters.minDistBetweenBlobs = 50.0         # 增大最大允许距离
        # parameters.minDistBetweenBlobs = 50.0
        # parameters.kmeansPP = True            # 使用 K-means++ 初始化
        # parameters.convexHullFactor = 1.2     # 放宽凸包因子
        # parameters.gridType = cv2.CIRCLES_GRID_ASYMMETRIC  # 非对称网格模式
        

        ret, centers = cv2.findCirclesGrid(gray, pattern_size, None, cv2.CALIB_CB_ASYMMETRIC_GRID, detector)

        if ret :
            objpoints.append(objp.astype(np.float32))
            # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # centers = cv2.cornerSubPix(gray, centers, (11, 11), (-1, -1), criteria)
            imgpoints.append(centers.astype(np.float32))

            # 可视化检测到的圆点
            if show_image:
                cv2.drawChessboardCorners(img, pattern_size, centers, ret)
                cv2.imshow('Detected Circles', img)
                cv2.waitKey(0)

    cv2.destroyAllWindows()

    # 鱼眼镜头标定
    DIM = gray.shape[::-1]  # 图像尺寸
    K = np.zeros((3, 3))  # 相机内参矩阵
    D = np.zeros((4, 1))  # 畸变系数
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(objpoints))]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(objpoints))]

    # 标定函数调用，使用正确的参数
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints, imgpoints, DIM, K, D,
        rvecs, tvecs,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

    all_error =[]
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints_proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints_proj, cv2.NORM_L2) / len(imgpoints_proj)
        all_error.append(error)
        mean_error += error
    print("Total Mean Reprojection Error: ", mean_error / len(objpoints))



"""




for fname in images_l:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (w,h), None)
    cv2.drawChessboardCorners(img, (w, h), corners, ret)
    cv2.imshow('FoundCorners', img)
    cv2.waitKey()
    if ret == True:
        objpoints.append(objp)
        corners2=cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints1.append(corners2)
ret, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints1, gray.shape[::-1],None,None)

# right camera calibration

for fname in images_r:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (w,h), None)
    cv2.drawChessboardCorners(img, (w, h), corners, ret)
    cv2.imshow('FoundCorners', img)
    cv2.waitKey(500)
    if ret == True:
        corners2=cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints2.append(corners2)

ret, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints2, gray.shape[::-1],None,None)

# binocular camera calibration
# flags = cv2.CALIB_FIX_INTRINSIC  # 固定内参，仅优化外参
ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints1, imgpoints2, mtx_l, dist_l, mtx_r, dist_r, gray.shape[::-1])

np.savez("parameters for calibration.npz",ret=ret,mtx_l=mtx_l,mtx_r=mtx_r,dist_l=dist_l,dist_r=dist_r,R=R,T=T)
np.savez("points.npz",objpoints=objpoints,imgpoints1=imgpoints1,imgpoints2=imgpoints2)

print('intrinsic matrix of left camera=\n', mtx_l)
print('intrinsic matrix of right camera=\n', mtx_r)
print('distortion coefficients of left camera=\n', dist_l)
print('distortion coefficients of right camera=\n', dist_r)
print('Transformation from left camera to right:\n')
print('R=\n', R)
print('\n')
print('T=\n', T)
print('\n')
print('Reprojection Error=\n', ret)

# stereo rectification
R1, R2, P1, P2, Q, ROI1, ROI2= cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, gray.shape[::-1], R, T, flags=0, alpha=-1)

# undistort rectifying mapping
map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, gray.shape[::-1], cv2.CV_16SC2)
map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, gray.shape[::-1], cv2.CV_16SC2)

# undistort the original image, take img#3 as an example
window_name = 'Image with Horizontal Lines'

# 创建一个可调整大小的窗口
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# 设置窗口的初始大小 (width, height)
desired_width = 1500  # 例如，设置宽度为800像素
desired_height = 1000 # 例如，设置高度为600像素
cv2.resizeWindow(window_name, desired_width, desired_height)
for i in range(0, len(img_dist_l)):

    left = cv2.imread(img_dist_l[i])
    dst_l = cv2.remap(left, map1_l, map2_l, cv2.INTER_LINEAR) #去除畸变以及垂直视差
    _, img_name = os.path.split(fname)
    cv2.imwrite(os.path.join(out_img_dist_l, img_name), dst_l)
    
    right = cv2.imread(img_dist_r[i])
    dst_r = cv2.remap(right, map1_r, map2_r, cv2.INTER_LINEAR)
    _, img_name = os.path.split(fname)
    cv2.imwrite(os.path.join(out_img_dist_r, img_name), dst_r)
    all = np.hstack((dst_l,dst_r))
    # 计算图像的高度和宽度
    height, width = all.shape[:2]

    # 计算五条水平线的位置
    num_lines = 5
    line_positions = [int(i * height / (num_lines + 1)) for i in range(1, num_lines + 1)]

    # 设置线条的颜色和厚度
    color = (0, 255, 0)  # 绿色线条
    thickness = 2

    # 在 all 图像上绘制水平线
    for pos in line_positions:
        cv2.line(all, (0, pos), (width, pos), color, thickness)
    cv2.imshow('Image with Horizontal Lines', all)
    cv2.waitKey(0)
cv2.destroyAllWindows()

np.savez("k.npz", mtx_l = mtx_l,mtx_r = mtx_l)"
"""