import cv2
import os

# 全局变量
refPt = []
cropping = False
roi_points = []

def click_and_crop(event, x, y, flags, param):
    global refPt, cropping, roi_points

    # 如果鼠标左键按下，记录起始点，并设置cropping为True
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # 鼠标移动时，如果正在裁剪，则更新结束点
    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        tmp_img = image.copy()
        cv2.rectangle(tmp_img, refPt[0], (x, y), (0, 255, 0), 2)
        # 调整图像大小以适应固定窗口
        tmp_img_resized = cv2.resize(tmp_img, (1280, 960))
        cv2.imshow("image", tmp_img_resized)

    # 如果鼠标左键释放，记录结束点，并将cropping设置为False
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False
        roi_points.append(refPt)

        # 在图像上绘制矩形框
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        # 调整图像大小以适应固定窗口
        image_resized = cv2.resize(image, (1280, 960))
        cv2.imshow("image", image_resized)

def crop_multiple_regions(image_path, output_dir):
    global image, refPt, cropping, roi_points
    image = cv2.imread(image_path)
    clone = image.copy()
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    # 设置窗口大小为固定1280*960
    cv2.resizeWindow("image", 1280, 960)
    cv2.setMouseCallback("image", click_and_crop)

    while True:
        # 调整图像大小以适应固定窗口
        image_resized = cv2.resize(image, (1280, 960))
        cv2.imshow("image", image_resized)
        key = cv2.waitKey(1) & 0xFF

        # 按r重置所有选择的区域
        if key == ord("r"):
            image = clone.copy()
            refPt = []
            cropping = False
            roi_points = []

        # 按c保存当前选中的区域并继续裁剪其他区域
        elif key == ord("c"):
            if len(roi_points) > 0:
                for i, pts in enumerate(roi_points):
                    # 需要根据原始图像和调整后图像的比例来计算实际裁剪区域
                    scale_x = clone.shape[1] / 1280.0
                    scale_y = clone.shape[0] / 960.0
                    
                    x1 = int(pts[0][0] * scale_x)
                    y1 = int(pts[0][1] * scale_y)
                    x2 = int(pts[1][0] * scale_x)
                    y2 = int(pts[1][1] * scale_y)
                    
                    # 确保坐标顺序正确
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    
                    roi = clone[y1:y2, x1:x2]
                    filename = os.path.splitext(os.path.basename(image_path))[0]
                    save_path = os.path.join(output_dir, f"{filename}_crop_{i}.jpg")
                    cv2.imwrite(save_path, roi)
                    print(f"图像已保存到 {save_path}")
            
        # 按q退出裁剪过程
        elif key == ord("q"):
            break
    
    cv2.destroyAllWindows()

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for img_name in os.listdir(input_dir):
        if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
            img_path = os.path.join(input_dir, img_name)
            crop_multiple_regions(img_path, output_dir)
            # 清空roi_points列表以便处理下一张图片
            roi_points.clear()

input_dir = '/home/erge/work/tools/minileft'
output_dir = '/home/erge/work/tools/cropleft'

process_images(input_dir, output_dir)