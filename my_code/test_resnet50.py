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
# 计算角误差损失函数
RAD2DEG = 180. / math.pi
def absolute_loss(pred, target):
    pred = pred.squeeze(dim=-1).squeeze(dim=-1)[:,:31]
    target = target[:,:31]

    numerator = torch.sum(pred * target, dim=1)  # shape: (N,)
    denominator = torch.sum(pred * pred, dim=1)  # shape: (N,)
    # 计算误差
    error = torch.norm(target - pred * numerator.unsqueeze(1) / denominator.unsqueeze(1), dim=1,
                       p=1)  # shape: (N,)
    return error.mean()
    # return np.linalg.norm(target - pred*np.dot(pred, target)/np.dot(pred, pred), ord=1)

def angular_loss(pred, target):#35.1754


    pred = pred.squeeze(dim=-1).squeeze(dim=-1)[:,:31]
    target = target[:,:31]
    # 归一化预测和目标
    pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True)+1e-9)
    target_norm = target / (torch.norm(target, dim=1, keepdim=True)+1e-9)
    # 计算角误差
    loss = torch.acos(torch.clamp(torch.sum(pred_norm * target_norm, dim=1), min=-1.0, max=1.0))
    return loss.mean()* RAD2DEG

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

# 指定要用到的设备
model = torch.nn.DataParallel(model, device_ids=device_ids)
# 模型加载到设备0
model = model.cuda(device=device_ids[0])

# 加载模型权重
model.load_state_dict(torch.load(model_path + 'epoch_3000.pth'))

# 数据加载器
dataloader_train, dataloader_test = load_data(batch_size, num_workers)

# 测试循环
model.eval()
with torch.no_grad():
    angular_error = 0.0
    absolute_error = 0.0
    for test_inputs, test_targets in dataloader_test:
        test_inputs, test_targets = test_inputs.cuda(device=device_ids[0]), test_targets.cuda(device=device_ids[0])
        test_output = model.forward(test_inputs)
        angular_error += angular_loss(test_output, test_targets).item()
        absolute_error += absolute_loss(test_output, test_targets).item()

angular_error /= len(dataloader_test)
absolute_error /= len(dataloader_test)
print("Angular Error: {:.4f}  Absolute Error: {:.4f}".format(angular_error, absolute_error))
