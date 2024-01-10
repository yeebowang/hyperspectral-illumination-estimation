clear;
% 加载数据
global data wavelength;
data = load('.\MatFiles_new_ds\s14_l1_I1.mat').data; % 替换为您的数据文件名
wavelength = load('.\wavelength\wavelength.mat').wavelength;

% RGB分量对应的索引

rgb_indices = [28, 18, 8];
% 提取RGB分量
R = mean(data(:,:,rgb_indices(1)),3);
G = mean(data(:,:,rgb_indices(2)),3);
B = mean(data(:,:,rgb_indices(3)),3);

% 归一化到0-1范围
double_R = double(R) ./ double(max(R(:)));
double_G = double(G) ./ double(max(G(:)));
double_B = double(B) ./ double(max(B(:)));

% 组合为RGB图像
rgb_image = cat(3, double_R, double_G, double_B);
% 显示RGB图像
figure;
imshow(rgb_image);
title('RGB Image');

set(gcf,'position',[0 0 900 600]);
% 注册点击事件
hFig = gcf;
hFig.WindowButtonDownFcn = @plotSpectralCurve;

% 定义全局变量以存储矩形框位置和点击次数
global clickCount rectPosition spectrumSource max_illum;

clickCount = 0;
rectPosition = [];
spectrumSource = [];

max_illum = 0.0;
function plotSpectralCurve(~, ~)
    global data wavelength;
    global clickCount rectPosition spectrumSource max_illum;
    
    clickCount = clickCount + 1;
    
    if clickCount == 1
        % 第1次点击，记录矩形框的起始点
        pt = get(gca, 'CurrentPoint');
        rectPosition(1, 1) = pt(1, 1);
        rectPosition(1, 2) = pt(1, 2);
    elseif clickCount == 2
        % 第2次点击，记录矩形框的结束点
        pt = get(gca, 'CurrentPoint');
        rectPosition(2, 1) = pt(1, 1);
        rectPosition(2, 2) = pt(1, 2);
% 计算矩形框内所有光谱点的平均值，作为光源光谱
        x_min = min(rectPosition(:, 1));
        x_max = max(rectPosition(:, 1));
        y_min = min(rectPosition(:, 2));
        y_max = max(rectPosition(:, 2));
        
        x_indices = round(x_min):round(x_max);
        y_indices = round(y_min):round(y_max);

        spectrumSource = squeeze(mean(mean(data(y_indices, x_indices, :), 1), 2));
        max_illum = max(max(spectrumSource));
        spectrumSource = spectrumSource / max(max(spectrumSource));
        disp('White reference area extracted.');
    elseif clickCount >= 3
        if mod(clickCount,2) == 1
            % 第1次点击，记录矩形框的起始点
            pt = get(gca, 'CurrentPoint');
            rectPosition(1, 1) = pt(1, 1);
            rectPosition(1, 2) = pt(1, 2);
        elseif mod(clickCount,2) == 0
            % 第2次点击，记录矩形框的结束点
            pt = get(gca, 'CurrentPoint');
            rectPosition(2, 1) = pt(1, 1);
            rectPosition(2, 2) = pt(1, 2);
            % 计算矩形框内所有光谱点的平均值
            x_min = min(rectPosition(:, 1));
            x_max = max(rectPosition(:, 1));
            y_min = min(rectPosition(:, 2));
            y_max = max(rectPosition(:, 2));
            
            x_indices = round(x_min):round(x_max);
            y_indices = round(y_min):round(y_max);
        
            % 第3次点击，绘制光谱曲线并除以光源光谱，然后归一化
    %         pt = get(gca, 'CurrentPoint');
    %         x = pt(1, 1);
    %         y = pt(1, 2);
    %         spectrum = double(squeeze(data(round(y), round(x), :)));        
            % 获取光谱曲线数据
            spectrum = squeeze(mean(mean(data(y_indices, x_indices, :), 1), 2));
            % 归一化到0~1范围
%             spectrum = spectrum  / max(max(spectrum));
            spectrum = spectrum  / max_illum;
            % 将光谱曲线除以光源光谱
            normalized_spectrum = spectrum ./ spectrumSource;
            normalized_spectrum = normalized_spectrum/ max(normalized_spectrum);
            % 绘制归一化后的光谱曲线
            figure;
            clf;
            
            hold on;

            plot(wavelength, squeeze(normalized_spectrum), 'b-','LineWidth',2);
            hold on;
            plot(wavelength, squeeze(spectrum), 'r:','LineWidth',2);
            hold on;
            plot(wavelength, squeeze(spectrumSource), 'k--','LineWidth',1.5);
    
            xlabel('Wavelength');
            ylabel('Normalized Intensity');
            title('Normalized Spectral Curve');
        end
    end
end