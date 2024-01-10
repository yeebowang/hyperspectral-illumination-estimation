clear;
% 设置文件夹路径
matFilesFolder = 'MatFiles_new_ds';
file_prefix = 'hsi/name_illum';
file_name = sprintf('%s.txt', file_prefix);
fileID = fopen(file_name, 'w');
annotationFolder = 'Anno';

% 获取.mat文件列表
matFiles = dir(fullfile(matFilesFolder, '*.mat'));

% 遍历每个.mat文件
for i = 1:length(matFiles)
    matFileName = matFiles(i).name;
    matFilePath = fullfile(matFilesFolder, matFileName);
%     outFilePath = fullfile(outFilesFolder, matFileName);
    % 获取名字的第一部分
    baseFileName = strsplit(matFileName, '_');
    baseFileName = baseFileName{1};
    
    restoreFileName = strsplit(matFileName, '.');
    restoreFileName = restoreFileName{1};
    % 从Annotation文件夹中读取对应的.png和.txt文件
    pngFilePath = fullfile(annotationFolder, [baseFileName, '.png']);
    txtFilePath = fullfile(annotationFolder, [baseFileName, '.txt']);
    
    % 读取.txt文件中的类别名称
    classFile = fopen(txtFilePath, 'r');
    classNames = textscan(classFile, '%s');
    fclose(classFile);
    classNames = classNames{1};
    
    % 读取.png文件中的分割GT
    segmentationGT = imread(pngFilePath);
    segmentationGT = flip(segmentationGT,2);
    % 创建一个与segmentationGT相同大小的全零矩阵
    mask = zeros(size(segmentationGT));
    
    % 遍历每个类别名称
    for j = 1:numel(classNames)
        class_indices = find(segmentationGT==j-1);  % 获取属于当前类别的像素索引
        if strcmp(classNames{j}, 'reference_white')
            
            mask(class_indices) = 0;  % 将属于当前类别的像素置为0，其他像素为1
        else
            mask(class_indices) = 1;
        end
    end
    load(matFilePath);  % 加载.mat文件中的data
    % 扩展mask为三维数组，以匹配data的维度
    mask_3d = repmat(mask, 1, 1, size(data, 3));
    mask_3d = logical(mask_3d); % 将 int16 类型的 mask_3d 转换为逻辑数组
    data = data(1:960,1:1056,:);
    mask_3d = mask_3d(1:960,1:1056,:);
    data(mask_3d) = 0;
    data = data(:,:,2:41);
    gt = mean(mean(data));
    gt = gt(1:34);
    gt = gt./max(gt);
    % 使用fprintf将数组中的数据写入文件，用空格隔开
    fprintf(fileID, '%s', restoreFileName);

    fprintf(fileID, ' %f', gt);
    
    % 写入换行符以分隔不同文件
    fprintf(fileID, '\n');
     
%     save(outFilePath, 'data');



end
