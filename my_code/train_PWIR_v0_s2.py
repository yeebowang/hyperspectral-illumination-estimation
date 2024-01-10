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
save_interval = 30
val_interval = 30
best_loss = 1e6
model_path = '../model_output/PWIR_v0_s2/'
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
# 训练循环
print('start training...')
for epoch in range(max_epochs):

    # 训练
    model.train()
    train_loss_ae = 0.0
    train_loss_abe = 0.0

    # 清零梯度
    optimizer.zero_grad()

    for batch_inputs, batch_targets, batch_names in dataloader_train:
        torch.cuda.empty_cache()
        batch_inputs, batch_targets = batch_inputs.cuda(device=device_ids[0]), batch_targets.cuda(device=device_ids[0])

        # 前向传播
        output_tensor = model.module.forward(batch_inputs)

        # 计算损失
        # output_L_mean = torch.mean(torch.mean(output_tensor, dim=2, keepdim=True), dim=3, keepdim=True)
        output_L_mean = output_tensor
        loss = angular_loss(output_L_mean, batch_targets)
        train_loss_ae += loss.item()
        train_loss_abe += absolute_loss(output_L_mean, batch_targets).item()
        # a = list(model.parameters())[0].clone()
        # 反向传播
        loss.backward()

        # 更新模型参数
        optimizer.step()
        # b = list(model.parameters())[0].clone()

        # for name, p in model.named_parameters():
        #     if p.requires_grad:
        #         # print(name)
        #         # p.apply(constraints)
        #         w = p.data
        #         w = w.clamp(0, 1)  # 将参数范围限制到-1-1之间
        #         p.data = w
        # 清零梯度
        optimizer.zero_grad()


    # 验证
    if (epoch + 1) % val_interval == 0 or epoch < val_interval:
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

        print_angular_loss(output_L_mean, val_targets)
        train_loss_ae /= len(dataloader_train)
        train_loss_abe /= len(dataloader_train)
        val_loss_ae /= len(dataloader_val)
        val_loss_abe /= len(dataloader_val)
        print("Epoch [{}/{}], Train ae: {:.4f}, Train abe: {:.4f}"
              "Val ae: {:.4f}, Val abe: {:.4f}"
              .format(epoch + 1, max_epochs, train_loss_ae, train_loss_abe,
                      val_loss_ae, val_loss_abe))
        if val_loss_ae < best_loss:
            best_loss = val_loss_ae
            torch.save(model.state_dict(), model_path + 'best.pth')
            print("Saved the best model at epoch", epoch+1)
    else:
        train_loss_ae /= len(dataloader_train)
        train_loss_abe /= len(dataloader_train)

        print("Epoch [{}/{}], Train ae: {:.4f}, Train abe: {:.4f}"

              .format(epoch + 1, max_epochs, train_loss_ae, train_loss_abe))
    # 保存模型
    if (epoch + 1) % save_interval == 0 or epoch < save_interval:
        torch.save(model.state_dict(), model_path + 'epoch_' + str(epoch+1).zfill(4) + '.pth')
