clear;
% 加载数据
% global data wavelength;
% data = load('.\MatFiles_new_ds\s4_l5_I5.mat').data; % 替换为您的数据文件名
wavelength = load('.\wavelength\wavelength.mat').wavelength;
figure;

hold on;
spectrumSource = load('.\illumination_spec\s4_l1_I5.mat').spectrumSource;
plot(wavelength, squeeze(spectrumSource), 'Color', [0.25, 0.5, 1],'LineWidth',2);
    
hold on;
spectrumSource = load('.\illumination_spec\s4_l2_I5.mat').spectrumSource;
plot(wavelength, squeeze(spectrumSource), 'Color', [0.25, 0.75, 0.75],'LineWidth',2);

hold on;
spectrumSource = load('.\illumination_spec\s4_l3_I5.mat').spectrumSource;
plot(wavelength, squeeze(spectrumSource), 'Color', [0.5, 0.5, 0.5],'LineWidth',2);

hold on;
spectrumSource = load('.\illumination_spec\s4_l4_I5.mat').spectrumSource;
plot(wavelength, squeeze(spectrumSource), 'Color', [0.75, 0.5, 0.25],'LineWidth',2);

hold on;
spectrumSource = load('.\illumination_spec\s4_l5_I5.mat').spectrumSource;
plot(wavelength, squeeze(spectrumSource), 'Color', [1, 0.75, 0],'LineWidth',2);

box on;
grid on;

xlabel('Wavelength');
ylabel('Normalized Intensity');
% title('Normalized Spectral Curve');
legend({'6500K','5500K', '4500K', '3500K','2500K'},'Location','northeast','FontSize',6)
