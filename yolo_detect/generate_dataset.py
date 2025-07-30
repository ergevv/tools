import os
from sklearn.model_selection import train_test_split
import shutil

# 设置文件夹路径
image_folder = "/home/erge/work/naruto/naruto20250702/frames"  # 图像文件夹路径
label_folder = "/home/erge/work/naruto/naruto20250702/label_txt"  # 标签文件夹路径
train_image_folder = "/home/erge/work/naruto/naruto20250702/naruto/images/train"  # 训练集图像保存路径
train_label_folder = "/home/erge/work/naruto/naruto20250702/naruto/labels/train"  # 训练集标签保存路径
val_image_folder = "/home/erge/work/naruto/naruto20250702/naruto/images/val"  # 验证集图像保存路径
val_label_folder = "/home/erge/work/naruto/naruto20250702/naruto/labels/val"  # 验证集标签保存路径


def find_matching_pairs(image_folder, label_folder):
    matched_pairs = []
    # 支持的图像扩展名列表
    supported_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
    
    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith(supported_extensions):  # 忽略大小写匹配扩展名
            label_file = os.path.splitext(image_file)[0] + ".txt"
            label_path = os.path.join(label_folder, label_file)
            if os.path.exists(label_path):
                matched_pairs.append(
                    (
                        os.path.join(image_folder, image_file),
                        label_path,
                    )
                )
    return matched_pairs


def split_dataset(matched_pairs, test_size=0.2):
    images, labels = zip(*matched_pairs)
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=test_size, random_state=42
    )
    return list(zip(train_images, train_labels)), list(zip(val_images, val_labels))


def move_files(pairs, dest_image_folder, dest_label_folder):
    os.makedirs(dest_image_folder, exist_ok=True)
    os.makedirs(dest_label_folder, exist_ok=True)
    for image_file, label_file in pairs:
        shutil.move(
            image_file, os.path.join(dest_image_folder, os.path.basename(image_file))
        )
        shutil.move(
            label_file, os.path.join(dest_label_folder, os.path.basename(label_file))
        )


# 查找匹配的图像-标签对
matched_pairs = find_matching_pairs(image_folder, label_folder)

# 分割数据集
train_pairs, val_pairs = split_dataset(matched_pairs)

# 移动或复制文件到相应的文件夹
move_files(train_pairs, train_image_folder, train_label_folder)
move_files(val_pairs, val_image_folder, val_label_folder)

print("处理完成！")
