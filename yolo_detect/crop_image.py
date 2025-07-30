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
        cv2.imshow("image", tmp_img)

    # 如果鼠标左键释放，记录结束点，并将cropping设置为False
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False
        roi_points.append(refPt)

        # 在图像上绘制矩形框
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

def crop_multiple_regions(image_path, output_dir):
    global image, refPt, cropping, roi_points
    image = cv2.imread(image_path)
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)

    while True:
        cv2.imshow("image", image)
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
                    roi = clone[pts[0][1]:pts[1][1], pts[0][0]:pts[1][0]]
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

input_dir = '/home/erge/work/naruto/frames'
output_dir = '/home/erge/work/naruto/crop_frames'

process_images(input_dir, output_dir)