import torch
import torch.nn as nn
import numpy as np
import random
from data_loader_256 import load_data
import os
import math
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.models as models
from model.cmf_net import CMFNET



# 设置超参数...

# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda")

# device_ids = [0, 1, 2, 3]  # 可用GPU
device_ids = [0]  # 可用GPU/
# batch_size = 1 * len(device_ids)
batch_size = 10
num_workers = 4
channel = 31

max_epochs = 1000
learning_rate = 1e-3
save_interval = 5
best_loss = 1e6
model_path = '../model_output/MIID_multi/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
seed = 42
random.seed(seed)
np.random.seed(seed)  # 0seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False  # Close optimization wyb
torch.backends.cudnn.deterministic = True

class SpectrumEstimationCNN(nn.Module):
    def __init__(self):
        super(SpectrumEstimationCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2), # add
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=8)
        )
        # 初始化权重
        for module in self.conv_layers.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # print('init...')
        # self.fc_layers = nn.Sequential(
        #     nn.Linear(128 * (height // 16) * (width // 16), 256),
        #     nn.ReLU(),
        #     nn.Linear(256, channel),
        #     # nn.Softmax(dim=1) #
        # )
    def forward(self, x):
        x = self.conv_layers(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc_layers(x)
        return x
# 创建模型实例
# model = SpectrumEstimationCNN()
model = CMFNET()
# 计算角误差损失函数...
# 计算角误差损失函数
RAD2DEG = 180. / math.pi
def angular_loss(pred, target):#35.1754

    # max_val, _ = torch.max(pred, dim=1)
    # min_val, _ = torch.min(pred, dim=1)
    # pred = pred.transpose(1,0)
    # pred = (pred - min_val)/(max_val - min_val)
    # pred = pred.transpose(1, 0)

    pred = pred.squeeze(dim=-1).squeeze(dim=-1)
    target = target[:,:31]

    # 归一化预测和目标
    pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True)+1e-9)
    target_norm = target / (torch.norm(target, dim=1, keepdim=True)+1e-9)
    # print('pred_norm', pred_norm)
    # print('target_norm',target_norm)
    # 计算角误差
    loss = torch.acos(torch.clamp(torch.sum(pred_norm * target_norm, dim=1), min=-1.0, max=1.0))
    return loss.mean()* RAD2DEG
def print_angular_loss(pred, target):#35.1754
    pred = pred.squeeze(dim=-1).squeeze(dim=-1)
    target = target[:,:31]
    # 归一化预测和目标
    pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True)+1e-9)
    target_norm = target / (torch.norm(target, dim=1, keepdim=True)+1e-9)

    print('pred_norm', pred_norm)
    print('target_norm',target_norm)
    # 计算角误差


# 创建SummaryWriter对象


# 指定要用到的设备...
# model = torch.nn.DataParallel(model, device_ids=device_ids)  # 0multi
# model = model.cuda(device=device_ids[0])  # 0multi

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


print('start training...')
# 训练循环...
for epoch in range(max_epochs):
    dataloader_train, dataloader_val = load_data(batch_size, num_workers)
    # 训练
    model.train()
    # model.eval()
    train_loss = 0.0

    # 清零梯度
    optimizer.zero_grad()

    iter_i = 0
    num_iter = len(dataloader_train)
    # with torch.no_grad():
    if 1:
        for batch_inputs, batch_targets in dataloader_train:

            if (iter_i*batch_size)%100==0:
                print("iter [{}/{}],".format(iter_i*batch_size, num_iter*batch_size))
            iter_i += 1
            # batch_inputs, batch_targets = batch_inputs.cuda(device=device_ids[0]), batch_targets.cuda(device=device_ids[0])  # 0multi

            # 前向传播...
            # ref_output, output_tensor = model.forward(batch_inputs, batch_inputs)
            output_tensor = model.forward(batch_inputs)
            # output_tensor = batch_inputs
            # print('line153')
            # 计算损失...
            output_tensor_mean = torch.mean(torch.mean(output_tensor, dim=2, keepdim=True), dim=3, keepdim=True)
            loss = angular_loss(output_tensor_mean, batch_targets)
            train_loss += loss.item()
            # print('line158')
            # 清零梯度...
            optimizer.zero_grad()
            # 反向传播...
            loss.backward()
            # print('line161')
            # 更新模型参数...
            optimizer.step()
            # print('line164')

            # if iter_i==num_iter:
            #     print_angular_loss(output_tensor, batch_targets)
            # del loss
            # del output_tensor
            # del output_tensor_mean
            # del batch_inputs
            # del batch_targets
            # torch.cuda.empty_cache() # this is the key code





    # 保存模型...
    if (epoch + 1) % save_interval == 0 or epoch == 0:
        # 验证...
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            iter_i = 0
            num_iter = len(dataloader_val)
            for val_inputs, val_targets in dataloader_val:
                if (iter_i * batch_size) % 100 == 0:
                    print("iter [{}/{}],".format(iter_i * batch_size, num_iter * batch_size))
                iter_i += 1
                # val_inputs, val_targets = val_inputs.cuda(device=device_ids[0]), val_targets.cuda(
                #     device=device_ids[0])  # 0multi
                # ref_output, val_output = model.forward(val_inputs, val_inputs)
                val_output = model.forward(val_inputs)
                val_output_mean = torch.mean(torch.mean(val_output, dim=2, keepdim=True), dim=3, keepdim=True)
                val_loss += angular_loss(val_output_mean, val_targets).item()

            del val_output
            del val_output_mean
            del val_inputs
            del val_targets
        train_loss /= len(dataloader_train)
        val_loss /= len(dataloader_val)
        print("Epoch [{}/{}], Train Loss: {:.4f}, Validation Loss: {:.4f}"
              .format(epoch + 1, max_epochs, train_loss, val_loss))

        torch.save(model.state_dict(), model_path + 'epoch_' + str(epoch + 1).zfill(4) + '.pth')
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_path + 'best.pth')
            print("Saved the best model at epoch", epoch + 1)
        del val_loss
    else:
        # 验证...
        train_loss /= len(dataloader_train)
        print("Epoch [{}/{}], Train Loss: {:.4f},"
              .format(epoch + 1, max_epochs, train_loss))

# 关闭SummaryWriter

