clc;
clear;
close all;

% 参数设置
rows = 4; % 圆点行数 (例如 4 行)
cols = 9; % 圆点列数 (例如 11 列)
spacing = 100; % 圆点间距 (单位：毫米)
skip_num = 1;
% 创建非对称圆网格的世界坐标
worldPoints = generateCircleGridPoints([rows, cols], spacing); %[x,y]

% FPS = 15
imageDir = fullfile('/home/erge/work/slam/kalibr_workspace/output3/cam0');

%% 单目
imageFiles = dir(fullfile(imageDir, '*.*g')); 
imagePaths = {imageFiles.name};


% 提取文件名中的数字部分
numbers = zeros(length(imagePaths), 1);
for i = 1:length(imagePaths)
    % 假设文件名格式为 "prefix<number>.png"，提取数字部分
    [~, name_without_ext] = fileparts(imagePaths{i});
    numbers(i) = str2double(regexp(name_without_ext, '\d+', 'match', 'once'));
end

% 按数字排序
[~, sort_idx] = sort(numbers);

% 获取排序后的文件列表
imagePaths = imagePaths(sort_idx);

% 初始化存储检测到的图像点
imagePoints = [];
times = [];

for i = 1:skip_num:length(imagePaths)
    imgPath = fullfile(imageDir, imagePaths{i});
    time = str2double(imagePaths{i}(1:end-4));
    % time = time*1000;
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

figure;
showReprojectionErrors(camParams);
title('重投影误差');

% 可视化外参
figure;
showExtrinsics(camParams);
title('外参可视化');

[intrinsicMatrix,distortionCoefficients] = cameraIntrinsicsToOpenCV(camParams);
% figure;
% showReprojectionErrors(camParams);
% title('标定重投影误差');
% 
% % 可视化外参
% figure;
% showExtrinsics(camParams);
% title('标定外参可视化');

%% 
% 构造YAML内容
yamlContent = sprintf(['cam0:\n', ...
    '  cam_overlaps: []\n', ...
    '  camera_model: pinhole\n', ...
    '  distortion_coeffs: [%f, %f, %f, %f]\n', ... % 使用实际畸变系数替换占位符
    '  distortion_model: radtan\n', ...
    '  intrinsics: [\n      %f,\n      %f,\n      %f,\n      %f,\n    ]\n', ... % 使用实际内参替换占位符
    '  resolution: [%d, %d]\n', ... % 使用图像分辨率替换占位符
    '  rostopic: /image'], ...
    distortionCoefficients(1), distortionCoefficients(2), distortionCoefficients(3), distortionCoefficients(4), ...
    intrinsicMatrix(1,1), intrinsicMatrix(2,2), intrinsicMatrix(1,3), intrinsicMatrix(2,3), ...
    size(I,2), size(I,1)); % 注意：size(I,2)是宽度，size(I,1)是高度

% 写入YAML文件
% yamlFilePath = fullfile('/path/to/save/', 'cam.yaml'); % 替换为您希望保存文件的实际路径
yamlFilePath = 'cam.yaml';
fid = fopen(yamlFilePath, 'w');
if fid == -1
    error('无法创建或写入YAML文件，请检查路径是否正确及是否有写权限。');
end
fprintf(fid, '%s', yamlContent);
fclose(fid);

disp('YAML文件已成功生成！');




%% 转为opencv格式并保存
% K = camParams.K;
% RadialDistortion = camParams.RadialDistortion;
% TangentialDistortion = camParams.TangentialDistortion;
imSize = size(I,1:2);
save("kalibr.mat","worldPoints","imagePoints","times","imSize");