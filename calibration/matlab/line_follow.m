clc;
clear;
close all;

% 参数设置
rows = 4; % 圆点行数 (例如 4 行)
cols = 9; % 圆点列数 (例如 11 列)
spacing = 100; % 圆点间距 (单位：毫米)

% 创建非对称圆网格的世界坐标
worldPoints = generateCircleGridPoints([rows, cols], spacing);


imageDir = fullfile('/home/erge/work/Aceii/calibration/photos/robot01/follow_line/20251027/left');
image_dir = "/home/erge/work/Aceii/calibration/photos/robot03/循线/20250828/rightFloor"; % floor
%% 单目
% 获取imageDir目录下所有的.jpg和.png文件
jpgFiles = dir(fullfile(imageDir, '*.*g'));
pngFiles = dir(fullfile(imageDir, '*.*g'));

% 合并两个结构体数组
imageFiles = [jpgFiles; pngFiles];
% 提取文件名中的数字（假设格式为 my_photo-<数字>.ext）
numList = zeros(length(imageFiles), 1);
for i = 1:length(imageFiles)
    % 使用正则表达式提取连字符和点之间的数字
    match = regexp(imageFiles(i).name, 'my_photo-(\d+)\.', 'tokens');
    if ~isempty(match)
        numList(i) = str2double(match{1}{1});
    else
        numList(i) = Inf; % 无法解析的放最后
    end
end

% 按提取出的数字从小到大排序
[~, sortedIdx] = sort(numList);

% 重新排列文件结构数组
imageFiles = imageFiles(sortedIdx);
imagePaths = {imageFiles.name};


% 初始化存储检测到的图像点
imagePoints = [];
ii = 1;
for i = 1:length(imagePaths)
    imgPath = fullfile(imageDir, imagePaths{i});
    I = imread(imgPath);
    % I = imrotate(I, -90, 'loose');

    % figure;
    % imshow(I);

    % 检测圆网格
    % detectedPoints = detectCircleGridPoints(I, [rows, cols],PatternType="symmetric");
    detectedPoints = detectCircleGridPoints(I, [rows, cols]);
    if ~isempty(detectedPoints) % 检查是否成功检测到圆网格
        fprintf('成功检测到圆网格：%s %i\n', imagePaths{i},ii);
        imagePoints= cat(3, imagePoints, detectedPoints); % 将检测到的点添加到 imagePoints
        ii=ii+1;
    else
        fprintf('未能检测到圆网格：%s\n', imagePaths{i});
    end
end


% 单目标定（分别标定左右相机）
[camParams, reprojectionErrors] = estimateCameraParameters(imagePoints, ...
    worldPoints, ...
    'ImageSize', size(I,1:2));% 可视化重投影误差
figure;
showReprojectionErrors(camParams);
title('标定重投影误差');

% 可视化外参
figure;
showExtrinsics(camParams);
title('标定外参可视化');







%% 转为opencv格式并保存
% K = camParams.K;
RadialDistortion = camParams.RadialDistortion;
TangentialDistortion = camParams.TangentialDistortion;
imSize = size(I,1:2);

[intrinsicMatrix,distortionCoefficients] = cameraIntrinsicsToOpenCV(camParams);
K = intrinsicMatrix;
save("resultSingle.mat","K","RadialDistortion", ...
    "TangentialDistortion","imSize");
distortion = distortionCoefficients;
cameraParams = camParams;

% 
% 
% % 标定板厚度4mm
plane_thickness = 4;
% 
% 
% 
% 读取相机参数
% K = mat_cam.K;
% RadialDistortion = mat_cam.RadialDistortion;
% TangentialDistortion = mat_cam.TangentialDistortion;




% 标定板参数
pattern_size = [7, 7];  % 角点行列数
square_size = 40;  % 单位：毫米


ext = "*.*g";
image_dir = fullfile(image_dir, ext);

% 生成标定板角点的世界坐标（Z=0）
zCoord = zeros(size(worldPoints,1),1);
obj_points = [worldPoints zCoord];

all_cam_points = [];  % 存储所有相机坐标系中的点

% 处理每张图像
image_paths = dir(image_dir);
for i = 1:length(image_paths)
    image_path = fullfile(image_paths(i).folder, image_paths(i).name);
    img = imread(image_path);
    % img = imrotate(img, -90, 'loose');
    gray = im2gray(img);
    % detectedPoints = detectCircleGridPoints(gray, [rows, cols],PatternType="symmetric");
    detectedPoints = detectCircleGridPoints(gray, [rows, cols]);

    if ~isempty(detectedPoints)
        % 求解PnP,Rwc
        [R,t] = extrinsics(detectedPoints,worldPoints,cameraParams);
        % 计算重投影误差

        reprojected_points = worldToImage(cameraParams, R, t, obj_points);
        error = mean(sqrt(sum((detectedPoints - reprojected_points).^2, 2)))
        
        % 转换到相机坐标系
        % R = R';
        % tvec = -R * t';

        % tvec = t;       
        % cam_points = (R * obj_points' + tvec')';
        cam_points = obj_points *R+t;
        all_cam_points = [all_cam_points; cam_points];
    end
end
% 
% % 平面拟合
centroid = mean(all_cam_points, 1);
A = all_cam_points - centroid;
[U, S, V] = svd(A);
n = V(:, 3)';  % 法向量 [a, b, c]
n = n / norm(n);  % 单位化
d = -dot(n, centroid);  % 平面方程参数 d

% 判断法向量方向，保证法向量与相机同向
if d < 0
    n = -n;
    d = -d;
end

d = d + plane_thickness * sqrt(n(1)^2 + n(2)^2 + n(3)^2);  % 去除板子厚度影响

% 平面方程: n(1)*X + n(2)*Y + n(3)*Z + d = 0
plane_equation = sprintf('%.6f * X + %.6f * Y + %.6f * Z + %.6f = 0', n(1), n(2), n(3), d);

% 计算夹角和距离
theta_rad = acos(n(3));  % 法向量与Z轴夹角
theta_deg = rad2deg(theta_rad);
distance = abs(d);  % 相机原点到平面的距离

fprintf('平面方程（相机坐标系）: %s\n', plane_equation);
fprintf('单位法向量: [%.6f, %.6f, %.6f]\n', n(1), n(2), n(3));
fprintf('与相机光轴的夹角: %.2f 度\n', theta_deg);
fprintf('到相机的垂直距离: %.3f 米\n', distance);

camera_name1 = "leftCamera";
camera_name2 = "rightCamera";
save_camera_config("line_follow.yaml", camera_name2, K, distortion, n, d)

function save_camera_config(filename, camera_name, K, distortion, plane_normal, d)
    % 打开文件用于写入
    fid = fopen(filename, 'w');
    if fid == -1
        error('无法打开文件 %s 进行写入', filename);
    end
    
    % 写入 YAML 头部
    fprintf(fid, '%%YAML:1.0\n\n');
    
    % 写入相机名称部分
    fprintf(fid, '%s:\n', camera_name);
    
    % 写入内参矩阵 K
    fprintf(fid, '  K:\n');
    fprintf(fid, '    - [%f, %f, %f]\n', K(1,1), K(1,2), K(1,3));
    fprintf(fid, '    - [%f, %f, %f]\n', K(2,1), K(2,2), K(2,3));
    fprintf(fid, '    - [%f, %f, %f]\n', K(3,1), K(3,2), K(3,3));
    
    % 写入畸变参数
    fprintf(fid, '  distortion: [%f, %f, %f, %f, %f]\n', ...
        distortion(1), distortion(2), distortion(3), distortion(4), distortion(5));
    
    % 写入平面方程参数
    fprintf(fid, '  plane_equation: [%f, %f, %f, %f]\n', ...
        plane_normal(1), plane_normal(2), plane_normal(3), d);
    
    % 关闭文件
    fclose(fid);
end

