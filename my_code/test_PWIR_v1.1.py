import torch
import torch.nn as nn
import numpy as np
import random
from data_loader import load_data
import os
import math
import torch.nn.functional as F
import cv2
import torchvision.models as models
import scipy.io as sio
from collections import OrderedDict
from argparse import ArgumentParser
parser = ArgumentParser(description='CMFNet')
parser.add_argument('--set', type=int, default=0, help='choose set id: 0,1,2')
args = parser.parse_args()
# 设置超参数
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
device_ids = [0, 1] # 可用GPU
batch_size = 16 * len(device_ids)
num_workers = 8
channel = 31
height = 512
width = 512
patch_size = 128 # should be more than 16(downsample rate)*4(last conv)=64
max_epochs = 3000
save_interval = 10
val_interval = 10
best_loss = 1e6
model_path = '../model_output/PWIR_v1_s2/'
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
def crop_into_patches(input_tensor, patch_size):
    batch_size, channel, height, width = input_tensor.shape
    patches = []
    stride = patch_size
    # 计算在 height 和 width 维度上的滑动次数
    num_vertical_patches = (height - patch_size) // stride + 1
    num_horizontal_patches = (width - patch_size) // stride + 1

    for i in range(num_vertical_patches):
        for j in range(num_horizontal_patches):
            # 计算裁剪窗口的起始位置
            start_h = i * stride
            start_w = j * stride
            # 在每个位置裁剪出小块数据
            patch = input_tensor[:, :, start_h:start_h+patch_size, start_w:start_w+patch_size]
            patches.append(patch)

    # 将所有裁剪出的小块数据拼接成新的张量
    # output_tensor = torch.cat(patches, dim=0)

    return patches
def gaussian_kernel(kernel_size, sigma):
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    gaussian_kernel = (1./(2.*np.pi*variance)) *\
                      torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) /\
                                (2*variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    return gaussian_kernel
def create_gaussian_kernel(sigma, num_channels):
    kernel_size = int(2*sigma) + 1
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    # kernel = kernel.expand(num_channels, num_channels, kernel_size, kernel_size)
    return kernel
def gaussian_blur(image, sigma):
    batch_size, num_channels, height, width = image.shape
    kernel = create_gaussian_kernel(sigma, num_channels).to(image.device)
    image_groups = torch.split(image, 1, dim=1)
    output_groups = []
    for group in image_groups:
        # 在这里进行每个组的计算
        output_i = F.conv2d(group, kernel, padding=kernel.shape[-1]//2)
        output_groups.append(output_i)

    # 将结果组合成一个张量，沿着dim=1维度连接回去
    output = torch.cat(output_groups, dim=1)

    return output
def triple_Gaussian(tensor):
    tensor_1 = gaussian_blur(tensor, 1.0)
    tensor_2 = gaussian_blur(tensor, 10.0)
    tensor_3 = gaussian_blur(tensor, 20.0)
    return torch.cat([tensor_1, tensor_2, tensor_3], dim=1)
def draw_triple_Gaussian(tensor):
    tensor_1 = gaussian_blur(tensor, 1.0)

    tensor_2 = gaussian_blur(tensor, 10.0)
    tensor_3 = gaussian_blur(tensor, 20.0)
    for i in range(tensor_3.shape[0]):
        img_1 = tensor_1[i, channel // 2, :, :].numpy()
        img_1 = img_1/img_1.max()*255
        img_2 = tensor_2[i, channel // 2, :, :].numpy()
        img_2 = img_2/img_2.max()*255
        img_3 = tensor_3[i, channel // 2, :, :].numpy()
        img_3 = img_3/img_3.max()*255
        cv2.imwrite('gauss_png/'+str(i).zfill(2) + '_gauss_1.png', img_1)
        cv2.imwrite('gauss_png/'+str(i).zfill(2) + '_gauss_10.png', img_2)
        cv2.imwrite('gauss_png/'+str(i).zfill(2) + '_gauss_20.png', img_3)

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
            nn.Conv2d(in_channels=channel*3, out_channels=32, kernel_size=5, stride=1, padding=2),
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


model = SpectrumEstimationResNet(num_channels=channel*3, spectrum_length=channel)
# 创建模型实例
# model = PWIR()


# 指定要用到的设备
model = torch.nn.DataParallel(model, device_ids=device_ids)
# 模型加载到设备0
model = model.cuda(device=device_ids[0])

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
dataloader_train, dataloader_test = load_data(batch_size, num_workers, args.set)
dataloader_val = dataloader_test
# 训练循环
print('start testing...')
model.load_state_dict(torch.load(model_path + 'epoch_0150.pth'))

model.eval()
# res_dir = '../prec_illum/PWIR_v1/'
res_dir = '../prec_illum/PWIR_v1/'

if not os.path.exists(res_dir):
    os.makedirs(res_dir)
with torch.no_grad():

    val_loss_ae = 0.0
    val_loss_abe = 0.0
    for val_inputs, val_targets, batch_names in dataloader_val:
        torch.cuda.empty_cache()
        patches = crop_into_patches(val_inputs, patch_size)
        for patch_i, patch in enumerate(patches, start=1):

            patch = patch.cuda(device=device_ids[0])
            patch_x3 = triple_Gaussian(patch)
            patch_x3, val_targets = patch_x3.cuda(device=device_ids[0]), val_targets.cuda(
                device=device_ids[0])
            # 前向传播
            output_tensor = model.module.forward(patch_x3)
            # output_L_mean = torch.mean(torch.mean(output_tensor, dim=2, keepdim=True), dim=3, keepdim=True)
            output_L_mean = output_tensor
            if patch_i ==1:
                output_L_sum = output_tensor
            else:
                output_L_sum = output_L_sum + output_tensor
            # val_loss_ae += angular_loss(output_L_mean, val_targets).item()
            # val_loss_abe += absolute_loss(output_L_mean, val_targets).item()
        output_L_sum = output_L_sum/patch_i
        val_loss_ae += angular_loss(output_L_sum, val_targets).item()
        val_loss_abe += absolute_loss(output_L_sum, val_targets).item()
        # save mat
        for b in range(len(batch_names)):

            pred = output_L_sum[b]
            # print(pred)
            prec_light = pred.detach().cpu().numpy()
            prec_light = prec_light[np.newaxis, :31, np.newaxis, np.newaxis]
            prec_light = np.tile(prec_light, (1, 1, 256, 256))
            print(prec_light.shape) # check for 1,31,256,256
            name = batch_names[b]
            sio.savemat(res_dir + 'res_' + name + '.mat', {'predict_light': prec_light})

print_angular_loss(output_L_mean, val_targets)
print(patch_i)

# val_loss_ae /= len(dataloader_val)*patch_i
# val_loss_abe /= len(dataloader_val)*patch_i
val_loss_ae /= len(dataloader_val)
val_loss_abe /= len(dataloader_val)
print(
      "Val ae: {:.4f}, Val abe: {:.4f}"
      .format(
              val_loss_ae, val_loss_abe))


