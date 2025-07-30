import cv2
import numpy as np
import scipy.io

# 加载 .mat 文件
mat_cam = scipy.io.loadmat("calibration/matlab/resultSingleFishEye.mat")

# 读取相机参数
MappingCoefficients = mat_cam["MappingCoefficients"][0]  # 映射系数
DistortionCenter = mat_cam["DistortionCenter"][0]  # 畸变中心
StretchMatrix = mat_cam["StretchMatrix"]  # 拉伸矩阵
image = cv2.imread("calibration/photos/single_fisheye/20250515/my_photo-6.jpg")
height, width = image.shape[:2]

image_size = (width, height)            # 图像分辨率

def theta_to_r(theta, a0, a2, a3, a4):
    """Scaramuzza 鱼眼模型：θ -> r"""
    return a0 + a2 * theta**2 + a3 * theta**3 + a4 * theta**4

def build_remap_tables(K_matrix, D_coeffs, image_size):
    """
    生成 remap 表格，用于去除鱼眼畸变。
    输入：
        K_matrix: 内参矩阵（包含 StretchMatrix 和 DistortionCenter）
        D_coeffs: [a0, a2, a3, a4]
        image_size: 图像尺寸 (w, h)
    输出：
        map_x, map_y: float32 类型，用于 cv2.remap()
    """
    a0, a2, a3, a4 = D_coeffs[0]
    w, h = image_size
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    cx, cy = K_matrix[0, 2], K_matrix[1, 2]

    for v in range(h):
        for u in range(w):
            dx = u - cx
            dy = v - cy
            r_pixel = np.sqrt(dx**2 + dy**2)
            if r_pixel == 0:
                theta = 0
            else:
                # 牛顿迭代法求解 theta，满足 r_pixel = a0 + a2*theta^2 + a3*theta^3 + a4*theta^4
                theta = 0.001
                for _ in range(10):
                    f = theta_to_r(theta, a0, a2, a3, a4) - r_pixel
                    df = 2*a2*theta + 3*a3*theta**2 + 4*a4*theta**3
                    theta -= f / df
            x_norm = theta * dx / r_pixel
            y_norm = theta * dy / r_pixel

            # 构造归一化坐标
            map_x[v, u] = x_norm
            map_y[v, u] = y_norm

    # 将归一化坐标映射回理想图像平面（使用 K_matrix 做线性变换）
    # 即：pixel = K @ [x_norm; y_norm; 1]
    ideal_coords = np.dstack([map_x, map_y])
    ideal_coords_homogeneous = np.dstack([ideal_coords, np.ones((h, w))])
    stretched_coords = np.einsum('ij,hwj->hwj', K_matrix, ideal_coords_homogeneous)[..., :2]
    map_x_ideal = stretched_coords[:, :, 0]
    map_y_ideal = stretched_coords[:, :, 1]

    return map_x_ideal, map_y_ideal

# 主程序入口
if __name__ == '__main__':
    # 构建内参矩阵 K = StretchMatrix + DistortionCenter
    K_matrix = np.array([
        [StretchMatrix[0][0], 0, DistortionCenter[0][0]],
        [0, StretchMatrix[1][1], DistortionCenter[0][1]],
        [0, 0, 1]
    ], dtype=np.float32)

    # 加载图像
    # image = cv2.imread("path_to_your_fisheye_image.jpg")

    # 构建 remap 表格
    map_x, map_y = build_remap_tables(K_matrix, MappingCoefficients, image_size)
    # 确保数据类型正确
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    # 应用 remap 进行去畸变
    undistorted_image = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    cv2.imshow("Original", image)
    cv2.imshow("Undistorted", undistorted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()