clc;
clear;
close all;

% 参数设置
rows = 7; % 圆点行数 (例如 4 行)
cols = 7; % 圆点列数 (例如 11 列)
spacing = 40; % 圆点间距 (单位：毫米)

% 创建非对称圆网格的世界坐标
worldPoints = generateCircleGridPoints([rows, cols], spacing,PatternType="symmetric");


imageDir = fullfile('/home/erge/python/Aceii/calibration/photos/single_fisheye/20250515');

%% 单目
calibrationImages = imageDatastore(imageDir);
imageFileNames = calibrationImages.Files;
[imagePoints,pairsUsed] = detectCircleGridPoints(imageFileNames,[rows, cols],PatternType="symmetric");

