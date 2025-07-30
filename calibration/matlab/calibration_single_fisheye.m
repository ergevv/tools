clc;
clear;
close all;

% 参数设置
rows = 7; % 圆点行数 (例如 4 行)
cols = 7; % 圆点列数 (例如 7 列)
spacing = 40; % 圆点间距 (单位：毫米)

% 创建非对称圆网格的世界坐标
worldPoints = generateCircleGridPoints([rows, cols], spacing,PatternType="symmetric");


imageDir = fullfile('/home/erge/python/Aceii/calibration/photos/single_fisheye/20250515');

%% 单目
imageFiles = dir(fullfile(imageDir, '*.jpg')); 
imagePaths = {imageFiles.name};

calibrationImages = imageDatastore(imageDir);
imageFileNames = calibrationImages.Files;

% 初始化存储检测到的图像点
imagePoints = [];

for i = 1:length(imagePaths)
    imgPath = fullfile(imageDir, imagePaths{i});
    I = imread(imgPath);
    
    % 检测圆网格
    detectedPoints = detectCircleGridPoints(I, [rows, cols],PatternType="symmetric");
    
    if ~isempty(detectedPoints) && size(detectedPoints,1)==rows*cols% 检查是否成功检测到圆网格
        fprintf('成功检测到圆网格：%s\n', imagePaths{i});
        imagePoints= cat(3, imagePoints, detectedPoints); % 将检测到的点添加到 imagePoints
        figure(1);
        imshow(I); hold on;
        plot(detectedPoints(:,1), detectedPoints(:,2), 'y+');
        hold off;
        title(imagePaths{i})
        pause();
        close;
        
    else
        fprintf('未能检测到圆网格：%s\n', imagePaths{i});
    end
end


% 单目标定
[camParams, reprojectionErrors] = estimateFisheyeParameters(imagePoints, ...
    worldPoints, ...
     [size(I,1) size(I,2)]);% 可视化重投影误差
figure;
showReprojectionErrors(camParams);
title('标定重投影误差');

% 可视化外参
figure;
showExtrinsics(camParams);
title('标定外参可视化');


J1 = undistortFisheyeImage(I,camParams.Intrinsics,'OutputView','full');
figure
imshowpair(I,J1,'montage')
title('Original Image (left) vs. Corrected Image (right)')

J2 = undistortFisheyeImage(I,camParams.Intrinsics,'OutputView','same', 'ScaleFactor', 0.5);
figure
imshow(J2)
title('Output View with low Scale Factor')

points = detectCheckerboardPoints(I);
[undistortedPoints,intrinsics1] = undistortFisheyePoints(points,camParams.Intrinsics);
[J, intrinsics2] = undistortFisheyeImage(I,camParams.Intrinsics,'OutputView','full'); %根据虚拟相机可得到归一化坐标








%% 转为opencv格式并保存
MappingCoefficients = camParams.Intrinsics.MappingCoefficients;
StretchMatrix = camParams.Intrinsics.StretchMatrix  ;
DistortionCenter = camParams.Intrinsics.DistortionCenter;
ImageSize = camParams.Intrinsics.ImageSize;
save("resultSingleFishEye.mat","MappingCoefficients","StretchMatrix", ...
     "DistortionCenter","ImageSize");