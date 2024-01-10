import torch
import torch.nn as nn
import numpy as np
import random
from data_loader_split import load_data
import os
import math
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.models as models
from model.cmf_net import CMFNET, get_Initial
# 设置超参数



# os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device_ids = [0,1] # 可用GPU
# batch_size = 1 * len(device_ids)
batch_size = 1
num_workers = 1
channel = 31
height = 512
width = 512
max_epochs = 3000
learning_rate = 1e-3
save_interval = 30
best_loss = 1e6
model_path = '../model_output/MIID/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
seed = 42
random.seed(seed)
np.random.seed(seed)  # 0seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


model = CMFNET()
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
# model = torch.nn.DataParallel(model, device_ids=device_ids) # 0multi
# 模型加载到设备0
# model = model.cuda(device=device_ids[0]) # 0multi

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
        torch.cuda.empty_cache()
        # batch_inputs, batch_targets = batch_inputs.cuda(device=device_ids[0]), batch_targets.cuda(device=device_ids[0]) # 0multi

        # 前向传播
        # output_tensor = model.module.forward(batch_inputs)
        # max_val, _ = torch.max(batch_inputs, dim=(1,2,3), keepdim=True, )
        # print(max_val)
        # batch_inputs = batch_inputs / max_val
        print(batch_inputs.shape)
        batch_initial = get_Initial(batch_inputs, 1, 'max')
        # test_x = torch.from_numpy(test_x).float()  # .to(device)
        # batch_initial = torch.from_numpy(batch_initial).float()  # .to(device)

        ref_output, output_tensor = model.forward(batch_inputs, batch_initial)
        # output_tensor = model.forward(batch_inputs)
        # min_val, _ = torch.min(output_tensor, dim=1)
        # print('min of output', min_val)
        # 计算损失
        output_tensor_mean = torch.mean(torch.mean(output_tensor, dim=2, keepdim=True), dim=3, keepdim=True)
        loss = angular_loss(output_tensor_mean, batch_targets)
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
                w = w.clamp(0, 1)  # 将参数范围限制到0-1之间
                p.data = w

        # 清零梯度
        optimizer.zero_grad()

    print_angular_loss(output_tensor, batch_targets)

    # 验证
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for val_inputs, val_targets in dataloader_val:
            # val_inputs, val_targets = val_inputs.cuda(device=device_ids[0]), val_targets.cuda(device=device_ids[0]) # 0multi
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
