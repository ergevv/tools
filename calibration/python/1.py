import yaml
import cv2
import numpy as np
import glob

# 加载 YAML 配置文件
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# 示例：加载配置并进行相机标定
if __name__ == "__main__":
    # 加载配置文件
    config = load_config('config.yaml')

    # 解析配置参数
    pattern_size = tuple(config['calibration']['pattern_size'])  # 标定板圆点的行列数
    square_size = config['calibration']['square_size']           # 圆点间距
    image_path = config['calibration']['image_path']             # 标定图像路径
    save_results = config['output']['save_results']              # 是否保存结果
    result_file = config['output']['result_file']                # 结果文件名

    print("标定板参数：")
    print(f"  pattern_size: {pattern_size}")
    print(f"  square_size: {square_size}")
    print(f"  image_path: {image_path}")
    print(f"  save_results: {save_results}")
    print(f"  result_file: {result_file}")

    # 准备对象点
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # 存储对象点和图像点的数组
    objpoints = []  # 3D 点在世界坐标系中的位置
    imgpoints = []  # 2D 点在图像平面中的位置

    # 加载标定图像
    images = glob.glob(image_path)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 寻找非对称圆标定板
        ret, corners = cv2.findCirclesGrid(gray, pattern_size, None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # 可视化检测到的圆点
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)
            cv2.imshow('Detected Circles', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # 相机标定
    if objpoints and imgpoints:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print("相机内参矩阵:\n", mtx)
        print("畸变系数:\n", dist)

        # 保存标定结果
        if save_results:
            np.savez(result_file, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
            print(f"标定结果已保存到 {result_file}")
    else:
        print("未能找到足够的标定点进行标定！")