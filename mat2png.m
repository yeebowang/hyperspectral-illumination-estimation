clear;
for i = [5,3,1,0]
    for l = 1:5
        for s = 1:18
            
            % 加载数据
            global data wavelength;
            data = load(['.\MatFiles_new_ds_out\s',num2str(s),'_l',num2str(l),'_I',num2str(i),'.mat']).data; % 替换为您的数据文件名
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
%             title('RGB Image');
            
            set(gcf,'position',[0 0 900 600]);
            saveas(gcf, ['.\PngFiles_mask\s',num2str(s),'_l',num2str(l),'_I',num2str(i),'.png']);
        end
    end
end