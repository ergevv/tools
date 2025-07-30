clc;
clear;
close all;

% 参数设置
rows = 10; % 圆点行数 (例如 4 行)
cols = 19; % 圆点列数 (例如 11 列)
spacing = 100; % 圆点间距 (单位：毫米)

% 创建非对称圆网格的世界坐标
worldPoints = generateCircleGridPoints([rows, cols], spacing);

date = '20250619';
show_result = true;
remove_outliers = false;

% 加载左右相机图像路径
leftSingleImageDir = fullfile('/home/erge/work/Aceii/calibration/photos', date, 'leftSingleImage'); % 替换为左相机图像文件夹路径
rightSingleImageDir = fullfile('/home/erge/work/Aceii/calibration/photos', date, '/rightSingleImage'); % 替换为右相机图像文件夹路径

leftStereoImageDir = fullfile('/home/erge/work/Aceii/calibration/photos', date,'/leftImage');
rightStereoImageDir = fullfile('/home/erge/work/Aceii/calibration/photos', date,'/rightImage');


%% 单目
leftImageFiles = dir(fullfile(leftSingleImageDir, '*.png'));
rightImageFiles = dir(fullfile(rightSingleImageDir, '*.png'));

leftImagePaths = {leftImageFiles.name};
rightImagePaths = {rightImageFiles.name};

% 初始化存储检测到的图像点
leftSingleImagePoints = [];
rightSingleImagePoints = [];





for i = 1:length(leftImagePaths)
    imgPath = fullfile(leftSingleImageDir, leftImagePaths{i});
    I = imread(imgPath);

    % 检测圆网格
    detectedPoints = detectCircleGridPoints(I, [rows, cols]);

    if ~isempty(detectedPoints) % 检查是否成功检测到圆网格
        fprintf('成功检测到圆网格：%s\n', leftImagePaths{i});
        leftSingleImagePoints= cat(3, leftSingleImagePoints, detectedPoints); % 将检测到的点添加到 imagePoints

    else
        fprintf('未能检测到圆网格：%s\n', leftImagePaths{i});
    end
end


for i = 1:length(rightImagePaths)
    imgPath = fullfile(rightSingleImageDir, rightImagePaths{i});
    I = imread(imgPath);

    % 检测圆网格
    detectedPoints = detectCircleGridPoints(I, [rows, cols]);

    if ~isempty(detectedPoints) % 检查是否成功检测到圆网格
        fprintf('成功检测到圆网格：%s\n', rightImagePaths{i});
        rightSingleImagePoints= cat(3, rightSingleImagePoints, detectedPoints); % 将检测到的点添加到 imagePoints
    else
        fprintf('未能检测到圆网格：%s\n', rightImagePaths{i});
    end
end


%% 立体标定
left_stereo_file = dir(fullfile(leftStereoImageDir, '*.png'));
right_stereo_file = dir(fullfile(rightStereoImageDir, '*.png'));

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
rightStereoImagePoints = [];
leftStereoImagePoints = [];
% Corresponding image name
image_name_stereo = [];
% 假设 left_stereo_name 和 right_stereo_name 的长度相同
for i = 1:length(left_stereo_name)
    % 读取左图像并检测圆网格
    leftImgPath = fullfile(leftStereoImageDir, left_stereo_name{i});
    leftI = imread(leftImgPath);
    leftDetectedPoints = detectCircleGridPoints(leftI, [rows, cols]);

    % 读取右图像并检测圆网格
    rightImgPath = fullfile(rightStereoImageDir, right_stereo_name{i});
    rightI = imread(rightImgPath);
    rightDetectedPoints = detectCircleGridPoints(rightI, [rows, cols]);

    % 检查左右图像是否都成功检测到圆网格
    if ~isempty(leftDetectedPoints) && ~isempty(rightDetectedPoints)
        % 如果左右图像都成功检测到圆网格，则将点加入
        fprintf('成功检测到圆网格：左图像 %s 和 右图像 %s\n', left_stereo_name{i}, right_stereo_name{i});
        leftStereoImagePoints = cat(3, leftStereoImagePoints, leftDetectedPoints); % 将左图像检测到的点加入
        rightStereoImagePoints = cat(3, rightStereoImagePoints, rightDetectedPoints); % 将右图像检测到的点加入
        % 使用正则表达式匹配数字
        numbers = regexp(left_stereo_name{i}, '\d+', 'match');

        % 将找到的数字从字符串转换为数值
        number = str2double(numbers{1});
        image_name_stereo = cat(3, image_name_stereo, number);
    else
        % 如果任意一个检测失败，则跳过，不加入任何点
        fprintf('未能同时检测到圆网格：左图像 %s 或 右图像 %s\n', left_stereo_name{i}, right_stereo_name{i});
    end
end





if show_result
    % 单目标定（分别标定左右相机）
    [leftCamParams, leftReprojectionErrors] = estimateCameraParameters(leftSingleImagePoints, ...
        worldPoints, ...
        'ImageSize', size(I,1:2));

    [rightCamParams, rightReprojectionErrors] = estimateCameraParameters(rightSingleImagePoints, ...
        worldPoints, ...
        'ImageSize', size(I,1:2));

    imagePoints= cat(4, leftStereoImagePoints, rightStereoImagePoints);

    % 标定双目外参
    [stereoParams, pairsUsed, ~] = estimateStereoBaseline(...
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

    if remove_outliers

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
        leftStereoImagePoints = leftStereoImagePoints( :, :,newPairsUsed);
        rightStereoImagePoints = rightStereoImagePoints( :, :,newPairsUsed);
        image_name_stereo = image_name_stereo(:,:,newPairsUsed);
        imagePoints= cat(4, imagePointsLeftGood, imagePointsRightGood);

        [stereoParams, pairsUsed, ~] = estimateStereoBaseline(...
            imagePoints, worldPoints, ...
            leftCamParams, rightCamParams);  %并没有固定内参

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
    end

end

imageSize = size(I,1:2);
save(date,"worldPoints","leftSingleImagePoints","rightSingleImagePoints", ...
    "leftStereoImagePoints","rightStereoImagePoints","imageSize","image_name_stereo");