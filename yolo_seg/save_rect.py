import cv2
import numpy as np
import glob
import os

screen_width = 960
screen_height = 600

image_dir = "/mnt/disk2/ubuntu/dataset/images"
ext = "*.png"
image_dir = os.path.join(image_dir, ext)
output_folder = "/mnt/disk2/ubuntu/dataset/rect"
os.makedirs(output_folder, exist_ok=True)
# 处理每张图像
image_paths = glob.glob(image_dir)
image_paths.sort()

rect_dir  = "/mnt/disk2/ubuntu/dataset/labels"

current_index = 0  # 当前图像索引

extend_rect = 10

def display_image(index):
    image_path = image_paths[index]
    
    # 获取文件名并改为txt后缀
    filename = os.path.basename(image_path)  
    name, _ = os.path.splitext(filename)     
    txt_filename = f"{name}.txt"             

    # 拼接rect_dir与txt文件名
    txt_filepath = os.path.join(rect_dir, txt_filename)

    # 读取文件内容
    if not os.path.exists(txt_filepath):
        print(f"文件 {txt_filepath} 不存在，跳过处理。")
        return None

    with open(txt_filepath, 'r') as file:
        content = file.read()

    lines = content.strip().split('\n')
    matrix = np.array([list(map(float, line.split())) for line in lines])

    if matrix.size == 0:
        print("文件为空，跳过处理。")
        return None

    mask = matrix[:, 0] == 2  # 创建布尔掩码，找出第一列等于 2 的行
    boxes = None

    if np.any(mask):  
        selected_rows = matrix[mask]          
        last_four_columns = selected_rows[:, 1:]  
        h, w = cv2.imread(image_path).shape[:2]

        boxes = last_four_columns.copy()
        boxes[:, 0] = (boxes[:, 0] - boxes[:, 2]/2) * w
        boxes[:, 1] = (boxes[:, 1] - boxes[:, 3]/2) * h
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2] * w
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3] * h
        boxes = boxes.astype(np.int32)
        print("image_path:",image_path)
    else:
        print("没有符合条件的数据。")

    img = cv2.imread(image_path)
    visual_img = img.copy()

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            x1 = max(0, x1 - extend_rect)
            y1 = max(0, y1 - extend_rect)
            x2 = min(img.shape[1], x2 + extend_rect)
            y2 = min(img.shape[0], y2 + extend_rect)
            # cv2.rectangle(visual_img, (x1, y1), (x2, y2), (0, 255, 0), 5)
            rect_img = img[y1:y2, x1:x2]
            original_filename = os.path.splitext(os.path.basename(image_path))[0]
            box_index = f"{x1}_{y1}_{x2}_{y2}"
            output_filename = f"{original_filename}_{box_index}.png"
            cv2.imwrite(os.path.join(output_folder, output_filename), rect_img)
            

num = len(image_paths)
while True:
    display_image(current_index)
    current_index +=1
    if current_index>=num:
        break


