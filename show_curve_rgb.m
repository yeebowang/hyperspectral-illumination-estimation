clear;

% 加载数据
wavelength = load('.\wavelength\wavelength.mat').wavelength(3:33);

% 加载spec数据
figure;
hold on;

spectrumSource = load('.\illumination_spec\s4_l5_I5.mat').spectrumSource(3:33);
spectrumSource = spectrumSource ./ max(spectrumSource);
plot(wavelength, squeeze(spectrumSource), 'k:', 'LineWidth', 2);
hold on;

% 加载RGB数据（假设文件中包含名为 'R', 'G', 'B' 的三个31x1的变量）
rgbData = load('rgb_31.mat');


% 计算R、G和B值
R_value = rgbData.r' * spectrumSource/100;
G_value = rgbData.g' * spectrumSource/100;
B_value = rgbData.b' * spectrumSource/100;
max_value = max([R_value, G_value, B_value]);
R_value = R_value / max_value;
G_value = G_value / max_value;
B_value = B_value / max_value;

% 绘制R、G和B值的折线图
wavelength_values = [450, 550, 650]; % 横坐标值
plot(wavelength_values, [B_value, G_value, R_value], 'k--', 'LineWidth', 2);
hold on;

% load pred illum
spectrumSource = load('.\illumination_spec\s4_l5_I1.mat').spectrumSource(3:33);
spectrumSource = spectrumSource ./ max(spectrumSource);
plot(wavelength, squeeze(spectrumSource),'Color', [0.25, 0.5, 1],  'LineWidth', 2);
hold on;
% 计算R、G和B值
R_value = rgbData.r' * spectrumSource/100;
G_value = rgbData.g' * spectrumSource/100;
B_value = rgbData.b' * spectrumSource/100;
max_value = max([R_value, G_value, B_value]);
R_value = R_value / max_value;
G_value = G_value / max_value;
B_value = B_value / max_value;
% 绘制R、G和B值的折线图
plot(wavelength_values, [B_value, G_value, R_value], 'Color', [1, 0.5, 0.25],  'LineWidth', 2);
hold on;
box on;
grid on;

xlabel('Wavelength (nm)');
ylabel('Intensity');
legend({'gt-spec','gt-rgb', 'pred-spec','pred-rgb'}, 'Location', 'northwest', 'FontSize', 6);
