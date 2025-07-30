import scipy.io
import numpy as np
import cv2

# ====================== 第一步：加载数据 ======================
# 加载 .mat 文件
# 加载 .mat 文件
mat_data = scipy.io.loadmat('calibration/calibration.mat')

# 提取数据
PatternPoints = mat_data['PatternPoints']
WorldPoints = mat_data['WorldPoints']

# 提取角点数据 (190, 2, 51)
image_points = PatternPoints

# 提取世界点数据 (190, 2)
world_points = WorldPoints

# 检查数据形状
print("角点数据形状：", image_points.shape)  # 输出 (190, 2, 51)
print("世界点数据形状：", world_points.shape)  # 输出 (190, 2)

# ====================== 第二步：准备标定输入 ======================
# 扩展世界点为 3D (190, 3)，Z 坐标设置为 0
world_points_3d = np.hstack([world_points, np.zeros((world_points.shape[0], 1))])

# 获取图像数量
num_images = image_points.shape[2]  # 图像数量 (51)

# 重复世界点以匹配每张图像 (51, 190, 3)
object_points = np.tile(world_points_3d, (num_images, 1, 1))

# 图像点列表 (51 个 (190, 2) 数组)
image_points_list = [image_points[:, :, i] for i in range(num_images)]

# 假设图像分辨率为 width x height
# 替换为实际图像分辨率
image_size = (1920, 1200)

# ====================== 第三步：执行相机标定 ======================
# 使用 OpenCV 进行相机标定
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objectPoints=object_points.astype(np.float32),  # 世界点 (51, 190, 3)
    imagePoints=image_points_list,                 # 图像点 (51 个 (190, 2) 数组)
    imageSize=image_size,                          # 图像分辨率
    cameraMatrix=None,                             # 初始化为 None
    distCoeffs=None                                # 初始化为 None
)

# 输出标定结果
print("\n相机内参矩阵：\n", camera_matrix)
print("\n畸变系数：\n", dist_coeffs)

# ====================== 第四步：计算重投影误差 ======================
mean_error = 0
for i in range(len(object_points)):
    # 重投影
    imgpoints_reprojected, _ = cv2.projectPoints(
        object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
    )
    # 计算误差
    error = cv2.norm(image_points_list[i], imgpoints_reprojected.reshape(-1, 2), cv2.NORM_L2)
    mean_error += error / len(object_points[i])

mean_error /= len(object_points)
print("\n平均重投影误差：", mean_error)