import torch
import torch.nn as nn
import numpy as np
import random
from data_loader import load_data
import os
import math
import torch.nn.functional as F
from collections import OrderedDict
# 设置超参数
device_ids = [0, 1] # 可用GPU
batch_size = 32 * len(device_ids)
# batch_size = 4 * len(device_ids)
channel = 34
height = 512
width = 512
max_epochs = 300
save_interval = 10
best_loss = 1e6
model_path = '../model_output/CNN/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
# seed = 42
# random.seed(seed)
# np.random.seed(seed)  # 0seed
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.benchmark = False  # Close optimization wyb
# torch.backends.cudnn.deterministic = True  # Close optimization wyb
class weightConstraint(object):
    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-1, 1)  # 将参数范围限制到-1-1之间
            module.weight.data = w
def get_model_norm_gradient(model):
    """
    Description:
        - get norm gradients from model, and store in a OrderDict

    Args:
        - model: (torch.nn.Module), torch model

    Returns:
        - grads in OrderDict
    """
    grads = OrderedDict()
    for name, params in model.named_parameters():
    # for name, params in model.parameters():
        print('name',name)
        grad = params.grad
        print('grad', grad)
        if grad is not None:
            grads[name] = grad.norm().item()
    return grads

def get_model_histogram(model):
    """
    Description:
        - get norm gradients from model, and store in a OrderDict

    Args:
        - model: (torch.nn.Module), torch model

    Returns:
        - grads in OrderDict
    """

    grads = OrderedDict()
    for name, params in model.named_parameters():
        grad = params.grad
        if grad is not None:
            tmp = {}
            params_np = grad.numpy()
            histogram, bins = np.histogram(params_np.flatten())
            tmp['histogram'] = list(histogram)
            tmp['bins'] = list(bins)
            grads[name] = tmp
    return grads

class SpectrumEstimationCNN(nn.Module):
    def __init__(self):
        super(SpectrumEstimationCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.AvgPool2d(kernel_size=2), # add
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=channel, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.AvgPool2d(kernel_size=16)
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
# 计算角误差损失函数
RAD2DEG = 180. / math.pi
def angular_loss(pred, target):#35.1754
    # pred = torch.tensor(pred, requires_grad=True)
    # target = torch.tensor(target, requires_grad=True)
    max_val, _ = torch.max(pred, dim=1)
    min_val, _ = torch.min(pred, dim=1)
    pred = pred.transpose(1,0)
    pred = (pred - min_val)/(max_val - min_val)
    pred = pred.transpose(1, 0)
    # print('min', torch.min(pred))

    # print('pred',pred, pred.shape)
    # print('target',target, target.shape)
    # 归一化预测和目标
    pred_norm = pred / torch.norm(pred, dim=1, keepdim=True)
    target_norm = target / torch.norm(target, dim=1, keepdim=True)
    # 计算角误差
    loss = torch.acos(torch.clamp(torch.sum(pred_norm * target_norm, dim=1), min=-1.0, max=1.0))
    return loss.mean()* RAD2DEG
    # ip = np.dot(pred, target)
    # norm = np.linalg.norm(pred) * np.linalg.norm(target)
    # return math.acos(ip/norm) * RAD2DEG

# 创建模型实例
model = SpectrumEstimationCNN()
# nn.init.kaiming_uniform_(model.weight, nonlinearity='relu')
# 指定要用到的设备
model = torch.nn.DataParallel(model, device_ids=device_ids)
# 模型加载到设备0
model = model.cuda(device=device_ids[0])

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
dataloader_train, dataloader_test = load_data(batch_size)
dataloader_val = dataloader_test
# 训练循环
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)
for epoch in range(max_epochs):

    # 训练
    model.train()
    train_loss = 0.0

    # 清零梯度
    optimizer.zero_grad()

    for batch_inputs, batch_targets in dataloader_train:
        batch_inputs, batch_targets = batch_inputs.cuda(device=device_ids[0]), batch_targets.cuda(device=device_ids[0])

        # 前向传播
        output_tensor = model.module.forward(batch_inputs)

        # 计算损失
        loss = angular_loss(output_tensor, batch_targets)
        train_loss += loss.item()
        a = list(model.parameters())[0].clone()
        # 反向传播
        loss.backward()

        # 更新模型参数
        optimizer.step()
        b = list(model.parameters())[0].clone()
        # print(output_tensor.grad)
        # print('if a=b:',torch.equal(a.data, b.data))
        # histo = (get_model_histogram(model))
        # print('histo',histo)
        # print(get_model_norm_gradient(model))
        # 清零梯度
        optimizer.zero_grad()



    # 验证
    model.eval()
    # with torch.no_grad():
    if 1:
        val_loss = 0.0
        for val_inputs, val_targets in dataloader_val:
            val_inputs, val_targets = val_inputs.cuda(device=device_ids[0]), val_targets.cuda(
                device=device_ids[0])
            val_output = model.forward(val_inputs)
            val_loss += angular_loss(val_output, val_targets).item()
            # val_loss += F.mse_loss(output_tensor, batch_targets)

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


# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)