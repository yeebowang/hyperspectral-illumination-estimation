import torch
import torch.nn as nn
import numpy as np
import random
from data_loader import load_data
import os
import math
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.models as models
# 设置超参数
device_ids = [0,1] # 可用GPU
batch_size = 16 * len(device_ids)
num_workers = 8
channel = 34
height = 512
width = 512
max_epochs = 3000
save_interval = 30
best_loss = 1e6
model_path = '../model_output/resnet50/'
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


model = SpectrumEstimationResNet(num_channels=34, spectrum_length=34)
# 计算角误差损失函数
RAD2DEG = 180. / math.pi
def angular_loss(pred, target):#35.1754

    # max_val, _ = torch.max(pred, dim=1)
    # min_val, _ = torch.min(pred, dim=1)
    # pred = pred.transpose(1,0)
    # pred = (pred - min_val)/(max_val - min_val)
    # pred = pred.transpose(1, 0)
    while(pred.shape!=target.shape):
            pred = pred.squeeze()

    # 归一化预测和目标
    pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True)+1e-9)
    target_norm = target / (torch.norm(target, dim=1, keepdim=True)+1e-9)
    # print('pred_norm', pred_norm)
    # print('target_norm',target_norm)
    # 计算角误差
    loss = torch.acos(torch.clamp(torch.sum(pred_norm * target_norm, dim=1), min=-1.0, max=1.0))
    return loss.mean()* RAD2DEG
def print_angular_loss(pred, target):#35.1754

    # max_val, _ = torch.max(pred, dim=1)
    # min_val, _ = torch.min(pred, dim=1)
    # pred = pred.transpose(1,0)
    # pred = (pred - min_val)/(max_val - min_val)
    # pred = pred.transpose(1, 0)
    while(pred.shape!=target.shape):
            pred = pred.squeeze()

    # 归一化预测和目标
    pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True)+1e-9)
    target_norm = target / (torch.norm(target, dim=1, keepdim=True)+1e-9)
    print('pred_norm', pred_norm)
    print('target_norm',target_norm)
    # 计算角误差
    loss = torch.acos(torch.clamp(torch.sum(pred_norm * target_norm, dim=1), min=-1.0, max=1.0))
    return loss.mean()* RAD2DEG



# 指定要用到的设备
model = torch.nn.DataParallel(model, device_ids=device_ids)
# 模型加载到设备0
model = model.cuda(device=device_ids[0])

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
dataloader_train, dataloader_test = load_data(batch_size, num_workers)
dataloader_val = dataloader_test
# 训练循环
# constraints=weightConstraint()
for epoch in range(max_epochs):

    # 训练
    model.train()
    train_loss = 0.0

    # 清零梯度
    optimizer.zero_grad()

    for batch_inputs, batch_targets in dataloader_train:
        batch_inputs, batch_targets = batch_inputs.cuda(device=device_ids[0]), batch_targets.cuda(device=device_ids[0])

        # 前向传播
        # output_tensor = model.module.forward(batch_inputs)
        output_tensor = model.forward(batch_inputs)
        min_val, _ = torch.min(output_tensor, dim=1)
        # print('min of output', min_val)
        # 计算损失
        loss = angular_loss(output_tensor, batch_targets)
        train_loss += loss.item()
        a = list(model.parameters())[0].clone()
        # 反向传播
        loss.backward()

        # 更新模型参数
        optimizer.step()
        b = list(model.parameters())[0].clone()

        for name, p in model.named_parameters():
            if p.requires_grad:
                # print(name)
                # p.apply(constraints)
                w = p.data
                w = w.clamp(0, 1)  # 将参数范围限制到-1-1之间
                p.data = w

        # 清零梯度
        optimizer.zero_grad()

    print_angular_loss(output_tensor, batch_targets)

    # 验证
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for val_inputs, val_targets in dataloader_val:
            val_inputs, val_targets = val_inputs.cuda(device=device_ids[0]), val_targets.cuda(device=device_ids[0])
            val_output = model.forward(val_inputs)
            val_loss += angular_loss(val_output, val_targets).item()


    train_loss /= len(dataloader_train)
    val_loss /= len(dataloader_val)
    print("Epoch [{}/{}], Train Loss: {:.4f}, Validation Loss: {:.4f}"
          .format(epoch+1, max_epochs, train_loss, val_loss))

    # 保存模型
    if (epoch + 1) % save_interval == 0:
        torch.save(model.state_dict(), model_path + 'epoch_' + str(epoch+1).zfill(4) + '.pth')
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_path + 'best.pth')
            print("Saved the best model at epoch", epoch+1)