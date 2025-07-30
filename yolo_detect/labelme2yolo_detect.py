import json
import os
import json




def convert_bbox(img_w, img_h, box):
    """
    Convert bounding box from [x1,y1,x2,y2] to YOLO format.
    """
    dw = 1.0 / img_w
    dh = 1.0 / img_h
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    width = box[2] - box[0]
    height = box[3] - box[1]
    x_center *= dw
    width *= dw
    y_center *= dh
    height *= dh
    return x_center, y_center, width, height


def json_to_yolo(json_file, output_file, class_list):
    with open(json_file, "r") as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]

    with open(output_file, "w") as out_f:
        for shape in data["shapes"]:
            if shape["shape_type"] != "rectangle":
                continue  # 忽略非矩形标注
            label = shape["label"]
            if label not in class_list:
                print(f"Label '{label}' not in class list. Skipping...")
                continue
            class_id = class_list.index(label)
            points = shape["points"]
            x1, y1 = points[0]
            x2, y2 = points[1]
            bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            yolo_bbox = convert_bbox(img_w, img_h, bbox)
            out_f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")


# 示例用法
# json_to_yolo('/home/erge/work/naruto/crop_datasets/labels', '/home/erge/work/naruto/crop_datasets/labels_detect', class_list)


json_dir = "/home/erge/work/naruto/naruto20250702/label_json"
label_dir = "/home/erge/work/naruto/naruto20250702/label_txt"
# 你的类别列表，顺序必须与训练一致
class_list = ["e"]  # 替换为你自己的类别

os.makedirs(label_dir, exist_ok=True)

for json_file in os.listdir(json_dir):
    if json_file.endswith(".json"):
        input_path = os.path.join(json_dir, json_file)
        output_path = os.path.join(label_dir, json_file.replace(".json", ".txt"))
        json_to_yolo(input_path, output_path, class_list)
