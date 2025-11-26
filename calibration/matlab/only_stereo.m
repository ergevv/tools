% clc;
% clear;
% close all;

% 参数设置
rows = 10; % 圆点行数 
cols = 19; % 圆点列数 
spacing = 100; % 圆点间距 (单位：毫米)

% 创建非对称圆网格的世界坐标
worldPoints = generateCircleGridPoints([rows, cols], spacing);

date = '20250805';


% 加载左右相机图像路径
leftImageDir = fullfile('/home/erge/work/Aceii/calibration/photos', date, 'leftSingleImage'); % 替换为左相机图像文件夹路径
rightImageDir = fullfile('/home/erge/work/Aceii/calibration/photos', date, '/rightSingleImage'); % 替换为右相机图像文件夹路径

left_stereo_path = fullfile('/home/erge/work/Aceii/calibration/photos', date,'/leftImage');
right_stereo_path = fullfile('/home/erge/work/Aceii/calibration/photos', date,'/rightImage');






%% 立体标定
left_stereo_file = dir(fullfile(left_stereo_path, '*.png')); 
right_stereo_file = dir(fullfile(right_stereo_path, '*.png'));




% 提取左文件名
file_names = {left_stereo_file.name};

% 提取文件名中的数字部分
numbers = zeros(length(file_names), 1);
for i = 1:length(file_names)
    % 假设文件名格式为 "prefix<number>.png"，提取数字部分
    [~, name_without_ext] = fileparts(file_names{i});
    numbers(i) = str2double(regexp(name_without_ext, '\d+', 'match', 'once'));
end

% 按数字排序
[~, sort_idx] = sort(numbers);

% 获取排序后的文件列表
left_stereo_file = left_stereo_file(sort_idx);


% 提取右文件名
file_names = {right_stereo_file.name};

% 提取文件名中的数字部分
numbers = zeros(length(file_names), 1);
for i = 1:length(file_names)
    % 假设文件名格式为 "prefix<number>.png"，提取数字部分
    [~, name_without_ext] = fileparts(file_names{i});
    numbers(i) = str2double(regexp(name_without_ext, '\d+', 'match', 'once'));
end

% 按数字排序
[~, sort_idx] = sort(numbers);

% 获取排序后的文件列表
right_stereo_file = right_stereo_file(sort_idx);

left_stereo_name = {left_stereo_file.name};
right_stereo_name = {right_stereo_file.name};




% 初始化存储检测到的图像点
rightImagePoints = [];
leftImagePoints = [];
% image_path
% 假设 left_stereo_name 和 right_stereo_name 的长度相同
for i = 1:length(left_stereo_name)
    % 读取左图像并检测圆网格
    leftImgPath = fullfile(left_stereo_path, left_stereo_name{i});
    leftI = imread(leftImgPath);
    if strcmp(left_stereo_name{i}, 'left956.png')
        leftI = imtranslate(leftI, [3, 0]);
    end
    leftDetectedPoints = detectCircleGridPoints(leftI, [rows, cols]);
    
    % 读取右图像并检测圆网格
    rightImgPath = fullfile(right_stereo_path, right_stereo_name{i});
    rightI = imread(rightImgPath);
    if strcmp(right_stereo_name{i}, 'right956.png')
        rightI = imtranslate(rightI, [-3, 0]);
    end
    rightDetectedPoints = detectCircleGridPoints(rightI, [rows, cols]);
    
    % 检查左右图像是否都成功检测到圆网格
    if ~isempty(leftDetectedPoints) && ~isempty(rightDetectedPoints)
        % 如果左右图像都成功检测到圆网格，则将点加入
        fprintf('成功检测到圆网格：左图像 %s 和 右图像 %s\n', left_stereo_name{i}, right_stereo_name{i});
        leftImagePoints = cat(3, leftImagePoints, leftDetectedPoints); % 将左图像检测到的点加入
        rightImagePoints = cat(3, rightImagePoints, rightDetectedPoints); % 将右图像检测到的点加入
    else
        % 如果任意一个检测失败，则跳过，不加入任何点
        fprintf('未能同时检测到圆网格：左图像 %s 或 右图像 %s\n', left_stereo_name{i}, right_stereo_name{i});
    end
end



imagePoints= cat(4, leftImagePoints, rightImagePoints);

[stereoParams2, pairsUsed, ReprojectionErrors] = estimateCameraParameters(imagePoints, ...
     worldPoints, 'ImageSize', size(leftI,1:2));

% 左右相机的初始内参矩阵和畸变系数
% camExtrinsics = estimateExtrinsics(imagePoints,worldPoints,intrinsics);

% 标定双目外参
% [stereoParams, pairsUsed, ReprojectionErrors] = estimateStereoBaseline(...
%     imagePoints, worldPoints, ...
%     leftCamParams, rightCamParams);  %并没有固定内参


disp('旋转矩阵（从左相机到右相机）：');
disp(stereoParams2.RotationOfCamera2);

disp('平移向量（从左相机到右相机）：');
disp(stereoParams2.TranslationOfCamera2);
disp(norm(stereoParams2.TranslationOfCamera2));

% 可视化重投影误差
figure;
showReprojectionErrors(stereoParams2);
title('立体标定重投影误差');

% 可视化外参
% figure;
% showExtrinsics(stereoParams2);
% title('立体标定外参可视化');
% 
% % 显示校正后的图像
% leftImgPath = fullfile(left_stereo_path, left_stereo_name{1});
% leftI = imread(leftImgPath);
% 
% rightImgPath = fullfile(right_stereo_path, right_stereo_name{1});
% rightI = imread(rightImgPath);
% [J1, J2] = rectifyStereoImages(leftI, rightI, stereoParams2,OutputView='full');
% figure;
% combinedImg = [J1, J2]; 
% imshow(combinedImg);
% title('Rectified Stereo Images');

% 假设 stereoParams 是你通过标定得到的立体相机参数结构体
leftImgPath = fullfile(left_stereo_path, left_stereo_name{1});
leftI = imread(leftImgPath);

rightImgPath = fullfile(right_stereo_path, right_stereo_name{1});
rightI = imread(rightImgPath);
[J1, J2] = rectifyStereoImages(leftI, rightI, stereoParams2,OutputView='full');

% 显示校正后的图像
figure;
combinedImg = [J1, J2]; 
imshow(combinedImg);
title('Rectified Stereo Images');
[intrinsicMatrix1,distortionCoefficients1,intrinsicMatrix2, ...
   distortionCoefficients2,rotationOfCamera2,translationOfCamera2] =... 
   stereoParametersToOpenCV(stereoParams2);

calibData.camera_left.K = intrinsicMatrix1;
calibData.camera_left.distortion = distortionCoefficients1;

calibData.camera_right.K = intrinsicMatrix2;
calibData.camera_right.distortion = distortionCoefficients2;

calibData.R = rotationOfCamera2;
calibData.T = translationOfCamera2;
fid = fopen('stereo_calibration.yaml', 'w');
fprintf(fid, '%%YAML:1.0\n');

% 写入 camera_left
fprintf(fid, 'camera_left:\n');
fprintf(fid, '  K:\n');
fprintf(fid, '    - [%f, %f, %f]\n', calibData.camera_left.K(1, :));
fprintf(fid, '    - [%f, %f, %f]\n', calibData.camera_left.K(2, :));
fprintf(fid, '    - [%f, %f, %f]\n', calibData.camera_left.K(3, :));

fprintf(fid, '  distortion: [%f, %f, %f, %f, %f]\n', calibData.camera_left.distortion);

% 写入 camera_right
fprintf(fid, 'camera_right:\n');
fprintf(fid, '  K:\n');
fprintf(fid, '    - [%f, %f, %f]\n', calibData.camera_right.K(1, :));
fprintf(fid, '    - [%f, %f, %f]\n', calibData.camera_right.K(2, :));
fprintf(fid, '    - [%f, %f, %f]\n', calibData.camera_right.K(3, :));

fprintf(fid, '  distortion: [%f, %f, %f, %f, %f]\n', calibData.camera_right.distortion);

% 写入 R 和 T
fprintf(fid, 'R:\n');
fprintf(fid, '  - [%f, %f, %f]\n', calibData.R(1, :));
fprintf(fid, '  - [%f, %f, %f]\n', calibData.R(2, :));
fprintf(fid, '  - [%f, %f, %f]\n', calibData.R(3, :));

fprintf(fid, 'T: [%f, %f, %f]\n', calibData.T(:)');

fclose(fid);