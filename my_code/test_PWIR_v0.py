import torch
import torch.nn as nn
import numpy as np
import random
from data_loader import load_data
import os
import math
import torchvision.models as models
import torch.nn.functional as F
from collections import OrderedDict
from argparse import ArgumentParser
import scipy.io as sio
parser = ArgumentParser(description='CMFNet')
parser.add_argument('--set', type=int, default=0, help='choose set id: 0,1,2')
args = parser.parse_args()
# 设置超参数
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
device_ids = [0,1] # 可用GPU
batch_size = 16 * len(device_ids)
num_workers = 8
channel = 31
height = 512
width = 512
max_epochs = 3000
save_interval = 10
val_interval = 10
best_loss = 1e6
model_path = '../model_output/PWIR_v0_s'+str(args.set)+'/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
seed = 42
random.seed(seed)
np.random.seed(seed)  # 0seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False  # Close optimization wyb
torch.backends.cudnn.deterministic = True  # Close optimization wyb

# 计算角误差损失函数
RAD2DEG = 180. / math.pi
def print_angular_loss(pred, target):
    pred = pred[0,:].squeeze().squeeze()
    target = target[0,:]
    pred_norm = pred / (torch.norm(pred)+1e-9)
    target_norm = target / (torch.norm(target)+1e-9)
    print('pred_norm', pred_norm)
    print('target_norm',target_norm)
def angular_loss(pred, target):#35.1754

    # max_val, _ = torch.max(pred, dim=1)
    # min_val, _ = torch.min(pred, dim=1)
    # pred = pred.transpose(1,0)
    # pred = (pred - min_val)/(max_val - min_val)
    # pred = pred.transpose(1, 0)
    pred = pred[:,:]
    target = target[:,:]

    # 归一化预测和目标
    pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True)+1e-9)
    target_norm = target / (torch.norm(target, dim=1, keepdim=True)+1e-9)
    # print('pred_norm', pred_norm)
    # print('target_norm',target_norm)
    # 计算角误差
    loss = torch.acos(torch.clamp(torch.sum(pred_norm * target_norm, dim=1), min=-1.0, max=1.0))
    return loss.mean()* RAD2DEG
def absolute_loss(pred, target):
    pred = pred[:,:]
    target = target[:,:]
    # 归一化预测和目标
    # pred = pred / (torch.norm(pred, dim=1, keepdim=True)+1e-9)
    # target = target / (torch.norm(target, dim=1, keepdim=True)+1e-9)

    numerator = torch.sum(pred * target, dim=1)+1e-9  # shape: (N,)
    denominator = torch.sum(pred * pred, dim=1)+1e-9  # shape: (N,)
    s = numerator / denominator

    # 计算误差
    error = torch.norm(target - pred * s.unsqueeze(1), dim=1,
                       p=1)  # shape: (N,)

    return error.mean()
class PWIR(nn.Module):
    def __init__(self):
        super(PWIR, self).__init__()
        self.conv_layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=channel, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # maxpool1
            nn.MaxPool2d(kernel_size=3, stride=2),
            # conv2
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # conv3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # avgpool1
            nn.AvgPool2d(kernel_size=3, stride=2),
            # conv4
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # avgpool2
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),

        )
        # 初始化权重
        for module in self.conv_layers.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)
                # print('init...')
        self.fc_layers = nn.Sequential(
            nn.Linear(channel, 256),
            nn.ReLU(),
            nn.Linear(256, channel),
            # nn.Softmax(dim=1) #
        )
        for module in self.fc_layers.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)
    def forward(self, x):
        x = self.conv_layers(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc_layers(x)
        return x


# 创建模型实例
class SpectrumEstimationResNet(nn.Module):
    def __init__(self, num_channels, spectrum_length, pretrained=True):
        super(SpectrumEstimationResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)

        # Modify the input layer to accept the desired number of channels
        self.resnet.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Adjust the size of the weight matrix in the FC layer
        self.fc = nn.Linear(1000, spectrum_length)

    def forward(self, x):
        features = self.resnet(x)
        features = features.view(features.size(0), -1)
        output = self.fc(features)
        return output


model = SpectrumEstimationResNet(num_channels=channel, spectrum_length=channel)
# model = PWIR()


# 指定要用到的设备
model = torch.nn.DataParallel(model, device_ids=device_ids)
# 模型加载到设备0
model = model.cuda(device=device_ids[0])

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
dataloader_train, dataloader_test = load_data(batch_size, num_workers, args.set)
dataloader_val = dataloader_test
model.load_state_dict(torch.load(model_path + 'epoch_0150.pth'))
res_dir = '../prec_illum/PWIR_v0/'
model.eval()
with torch.no_grad():

    val_loss_ae = 0.0
    val_loss_abe = 0.0
    for val_inputs, val_targets, batch_names in dataloader_val:
        torch.cuda.empty_cache()
        val_inputs, val_targets = val_inputs.cuda(device=device_ids[0]), val_targets.cuda(
            device=device_ids[0])
        val_output = model.forward(val_inputs)
        # output_L_mean = torch.mean(torch.mean(val_output, dim=2, keepdim=True), dim=3, keepdim=True)
        output_L_mean = val_output
        val_loss_ae += angular_loss(output_L_mean, val_targets).item()
        val_loss_abe += absolute_loss(output_L_mean, val_targets).item()

        # save mat
        for b in range(len(batch_names)):

            pred = output_L_mean[b]
            # print(pred)
            prec_light = pred.detach().cpu().numpy()
            prec_light = prec_light[np.newaxis, :31, np.newaxis, np.newaxis]
            prec_light = np.tile(prec_light, (1, 1, 256, 256))
            print(prec_light.shape) # check for 1,31,256,256
            name = batch_names[b]
            sio.savemat(res_dir + 'res_' + name + '.mat', {'predict_light': prec_light})

print_angular_loss(output_L_mean, val_targets)

val_loss_ae /= len(dataloader_val)
val_loss_abe /= len(dataloader_val)
print(
      "Val ae: {:.4f}, Val abe: {:.4f}"
      .format(
              val_loss_ae, val_loss_abe))


