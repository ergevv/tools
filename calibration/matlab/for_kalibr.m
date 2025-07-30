clc;
clear;
close all;

% 参数设置
rows = 10; % 圆点行数 (例如 4 行)
cols = 19; % 圆点列数 (例如 11 列)
spacing = 100; % 圆点间距 (单位：毫米)

% 创建非对称圆网格的世界坐标
worldPoints = generateCircleGridPoints([rows, cols], spacing); %[x,y]


imageDir = fullfile('/home/erge/work/slam/kalibr_workspace/output5/cam0');

%% 单目
imageFiles = dir(fullfile(imageDir, '*.png')); 
imagePaths = {imageFiles.name};


% 初始化存储检测到的图像点
imagePoints = [];
times = [];

for i = 1:length(imagePaths)
    imgPath = fullfile(imageDir, imagePaths{i});
    time = str2double(imagePaths{i}(1:end-4));
    I = imread(imgPath);
    
    % 检测圆网格
    detectedPoints = detectCircleGridPoints(I, [rows, cols]);
    
    if ~isempty(detectedPoints) % 检查是否成功检测到圆网格
        fprintf('成功检测到圆网格：%s\n', imagePaths{i});
         imagePoints= cat(3, imagePoints, detectedPoints); % 将检测到的点添加到 imagePoints[x,y]
         times = cat(2,times,time);
        
    else
        fprintf('未能检测到圆网格：%s\n', imagePaths{i});
    end
end


% % 单目标定（分别标定左右相机）
[camParams, reprojectionErrors] = estimateCameraParameters(imagePoints, ...
    worldPoints, ...
    'ImageSize', size(I,1:2));% 可视化重投影误差
% figure;
% showReprojectionErrors(camParams);
% title('标定重投影误差');
% 
% % 可视化外参
% figure;
% showExtrinsics(camParams);
% title('标定外参可视化');

%% 转为opencv格式并保存
% K = camParams.K;
% RadialDistortion = camParams.RadialDistortion;
% TangentialDistortion = camParams.TangentialDistortion;
imSize = size(I,1:2);
save("kalibr.mat","worldPoints","imagePoints","times","imSize");