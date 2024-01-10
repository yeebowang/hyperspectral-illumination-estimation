% r = r./max(r);
% g = g./max(g);
% b = b./max(b);
clear;

% 加载数据
wavelength = load('.\wavelength\wavelength.mat').wavelength(3:33);
% 加载RGB数据（假设文件中包含名为 'R', 'G', 'B' 的三个31x1的变量）
rgbData = load('rgb_31.mat');
load('GaiaSkyMini2.mat');
figure;
hold on;
hold on; plot(wavelength,rgbData.r,'r-','LineWidth',3);% draw pred
hold on; plot(wavelength,rgbData.g,'g-','LineWidth',3);% draw pred
hold on; plot(wavelength,rgbData.b,'b-','LineWidth',3);% draw pred
value = YSignalValue;
value = value ./ max(value) .* 75;
hold on; plot(XWaveLengthNm,value,'k-','LineWidth',3);% draw pred

box on;
grid on;
legend({'RedFilter','GreenFilter','BlueFilter', 'HyperSpec'},'Location','northeast','FontSize',6)
xlim([400 1000]);
xlabel('Wavelength (nm)');
ylabel('Quantum Efficiency (%)');


