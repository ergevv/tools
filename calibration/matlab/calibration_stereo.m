clc;
clear;
close all;

% 参数设置
rows = 10; % 圆点行数 (例如 4 行)
cols = 19; % 圆点列数 (例如 11 列)
spacing = 100; % 圆点间距 (单位：毫米)

% 创建非对称圆网格的世界坐标
worldPoints = generateCircleGridPoints([rows, cols], spacing);

date = '2025072302';


% 加载左右相机图像路径
leftImageDir = fullfile('/home/erge/work/Aceii/calibration/photos', date, 'leftSingleImage'); % 替换为左相机图像文件夹路径
rightImageDir = fullfile('/home/erge/work/Aceii/calibration/photos', date, '/rightSingleImage'); % 替换为右相机图像文件夹路径

left_stereo_path = fullfile('/home/erge/work/Aceii/calibration/photos', date,'/leftImage');
right_stereo_path = fullfile('/home/erge/work/Aceii/calibration/photos', date,'/rightImage');









%% 单目
leftImageFiles = dir(fullfile(leftImageDir, '*.png')); 
rightImageFiles = dir(fullfile(rightImageDir, '*.png'));

leftImagePaths = {leftImageFiles.name};
rightImagePaths = {rightImageFiles.name};

% 初始化存储检测到的图像点
rightImagePoints = [];
leftImagePoints = [];

for i = 1:length(leftImagePaths)
    imgPath = fullfile(leftImageDir, leftImagePaths{i});
    I = imread(imgPath);
    
    % 检测圆网格
    detectedPoints = detectCircleGridPoints(I, [rows, cols]);
    
    if ~isempty(detectedPoints) % 检查是否成功检测到圆网格
        fprintf('成功检测到圆网格：%s\n', leftImagePaths{i});
         leftImagePoints= cat(3, leftImagePoints, detectedPoints); % 将检测到的点添加到 imagePoints
        
    else
        fprintf('未能检测到圆网格：%s\n', leftImagePaths{i});
    end
end


for i = 1:length(rightImagePaths)
    imgPath = fullfile(rightImageDir, rightImagePaths{i});
    I = imread(imgPath);
    
    % 检测圆网格
    detectedPoints = detectCircleGridPoints(I, [rows, cols]);
    
    if ~isempty(detectedPoints) % 检查是否成功检测到圆网格
        fprintf('成功检测到圆网格：%s\n', rightImagePaths{i});
         rightImagePoints= cat(3, rightImagePoints, detectedPoints); % 将检测到的点添加到 imagePoints
    else
        fprintf('未能检测到圆网格：%s\n', rightImagePaths{i});
    end
end


% 单目标定（分别标定左右相机）
[leftCamParams, leftReprojectionErrors] = estimateCameraParameters(leftImagePoints, ...
    worldPoints, ...
    'ImageSize', size(I,1:2));

[rightCamParams, rightReprojectionErrors] = estimateCameraParameters(rightImagePoints, ...
    worldPoints, ...
    'ImageSize', size(I,1:2));

% 可视化重投影误差
figure;
showReprojectionErrors(leftCamParams);
title('左目标定重投影误差');

% 可视化外参
figure;
showExtrinsics(leftCamParams);
title('左目标定外参可视化');

% 可视化重投影误差
figure;
showReprojectionErrors(rightCamParams);
title('右目标定重投影误差');

% 可视化外参
figure;
showExtrinsics(rightCamParams);
title('右目标定外参可视化');



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
    leftDetectedPoints = detectCircleGridPoints(leftI, [rows, cols]);
    
    % 读取右图像并检测圆网格
    rightImgPath = fullfile(right_stereo_path, right_stereo_name{i});
    rightI = imread(rightImgPath);
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
% 
% [stereoParams, pairsUsed, ReprojectionErrors] = estimateCameraParameters(imagePoints, ...
%      worldPoints, 'ImageSize', size(I,1:2));

% 左右相机的初始内参矩阵和畸变系数
% camExtrinsics = estimateExtrinsics(imagePoints,worldPoints,intrinsics);

% 标定双目外参
[stereoParams, pairsUsed, ReprojectionErrors] = estimateStereoBaseline(...
    imagePoints, worldPoints, ...
    leftCamParams, rightCamParams);  %并没有固定内参



% 显示标定结果
disp('左相机内参矩阵：');
disp(stereoParams.CameraParameters1.IntrinsicMatrix);
disp(leftCamParams.IntrinsicMatrix);

disp('右相机内参矩阵：');
disp(stereoParams.CameraParameters2.IntrinsicMatrix);
disp(rightCamParams.IntrinsicMatrix);


disp('径向畸变系数（左相机）：');
disp(stereoParams.CameraParameters1.RadialDistortion);
disp(leftCamParams.RadialDistortion);


disp('径向畸变系数（右相机）：');
disp(stereoParams.CameraParameters2.RadialDistortion);
disp(rightCamParams.RadialDistortion);

disp('旋转矩阵（从左相机到右相机）：');
disp(stereoParams.RotationOfCamera2);

disp('平移向量（从左相机到右相机）：');
disp(stereoParams.TranslationOfCamera2);
disp(norm(stereoParams.TranslationOfCamera2));

% 可视化重投影误差
figure;
showReprojectionErrors(stereoParams);
title('立体标定重投影误差');

% 可视化外参
figure;
showExtrinsics(stereoParams);
title('立体标定外参可视化');

% % 4. 剔除误差过大的帧，使用了RANSAC，不一定需要这一步
% errorThreshold = 0.3;
% left_error = stereoParams.CameraParameters1.ReprojectionErrors; 
% reight_error = stereoParams.CameraParameters2.ReprojectionErrors; 

% 获取实际用于标定的帧索引
usedIndices = find(pairsUsed);

% 提取左右相机重投影误差
errorsLeft = stereoParams.CameraParameters1.ReprojectionErrors;
errorsRight = stereoParams.CameraParameters2.ReprojectionErrors;
% 计算每帧的平均误差（左右相机平均）
meanErrors = zeros(numel(usedIndices), 1);
for i = 1:numel(usedIndices)
    idx = usedIndices(i);
    % 左相机误差
    errL = squeeze(errorsLeft(:, :,idx));
    rmseL = mean(sqrt(sum(errL.^2, 2)));
    % 右相机误差
    errR = squeeze(errorsRight( :, :,idx));
    rmseR = mean(sqrt(sum(errR.^2, 2)));
    meanErrors(i) = (rmseL + rmseR) / 2;
end
% 设定阈值（例如均值+2倍标准差）
meanError = mean(meanErrors);
stdError = std(meanErrors);
threshold = meanError + 2 * stdError;

% 找到高误差帧的索引
badInUsed = find(meanErrors > threshold);
badIndices = usedIndices(badInUsed);

% 创建新索引，排除高误差帧
newPairsUsed = pairsUsed;
newPairsUsed(badIndices) = false;

% 提取有效角点数据
imagePointsLeftGood = leftImagePoints( :, :,newPairsUsed);
imagePointsRightGood = rightImagePoints( :, :,newPairsUsed);

imagePoints= cat(4, imagePointsLeftGood, imagePointsRightGood);

% [stereoParams, pairsUsed, estimationErrors] = estimateCameraParameters(imagePoints, ...
%     worldPoints, 'ImageSize', size(I,1:2));
[stereoParams, pairsUsed, ReprojectionErrors] = estimateStereoBaseline(...
    imagePoints, worldPoints, ...
    leftCamParams, rightCamParams); 


disp('平移向量（从左相机到右相机）：');
disp(stereoParams.TranslationOfCamera2);
disp(norm(stereoParams.TranslationOfCamera2));

% 可视化重投影误差
figure;
showReprojectionErrors(stereoParams);
title('立体标定重投影误差');

% 可视化外参
figure;
showExtrinsics(stereoParams);
title('立体标定外参可视化');

% 假设 stereoParams 是你通过标定得到的立体相机参数结构体
leftImgPath = fullfile(left_stereo_path, left_stereo_name{1});
leftI = imread(leftImgPath);

rightImgPath = fullfile(right_stereo_path, right_stereo_name{1});
rightI = imread(rightImgPath);
[J1, J2] = rectifyStereoImages(leftI, rightI, stereoParams,OutputView='full');

% 显示校正后的图像
figure;
combinedImg = [J1, J2]; 
imshow(combinedImg);
title('Rectified Stereo Images');

% % 2. 立体校正
% [J1, J2] = rectifyStereoImages(leftI, rightI, stereoParams);
% 
% % 3. 显示校正结果（可选）
% figure;
% imshow(stereoAnaglyph(J1, J2)); % 红蓝图查看对齐效果
% title('Rectified Frames');

%% 转为opencv格式并保存
[intrinsicMatrix1,distortionCoefficients1,intrinsicMatrix2, ...
   distortionCoefficients2,rotationOfCamera2,translationOfCamera2] =... 
   stereoParametersToOpenCV(stereoParams);

save("result.mat","translationOfCamera2","rotationOfCamera2", ...
    "distortionCoefficients2","intrinsicMatrix2","distortionCoefficients1","intrinsicMatrix1");


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



% 创建 Excel 文件
filename = [date,'.xlsx'];

% 写入每个参数到单独的工作表



writematrix(intrinsicMatrix1, filename, 'Sheet', 'IntrinsicMatrix1');
writematrix(distortionCoefficients1, filename, 'Sheet', 'DistortionCoefficients1');
writematrix(intrinsicMatrix2, filename, 'Sheet', 'IntrinsicMatrix2');
writematrix(distortionCoefficients2, filename, 'Sheet', 'DistortionCoefficients2');
writematrix(rotationOfCamera2, filename, 'Sheet', 'RotationOfCamera2');
writematrix(translationOfCamera2, filename, 'Sheet', 'TranslationOfCamera2');

disp(['数据已成功保存到 ', filename]);