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
from model.cmf_net import CMFNET, get_Initial
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter

# 设置超参数...

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda")

device_ids = [0, 1, 2, 3]  # 可用GPU
# batch_size = 1 * len(device_ids)
batch_size = 4
num_workers = 4
channel = 31

max_epochs = 1000
learning_rate = 1e-3
save_interval = 10
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
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


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
writer = SummaryWriter(log_dir="logs")

# 指定要用到的设备...
model = torch.nn.DataParallel(model, device_ids=device_ids)  # 0multi
model = model.cuda(device=device_ids[0])  # 0multi

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
dataloader_train, dataloader_val = load_data(batch_size, num_workers)
dataloader_train = dataloader_train
dataloader_val = dataloader_val
print('start training...')
# 训练循环...
for epoch in range(max_epochs):
    # 清零梯度...
    optimizer.zero_grad()

    # 使用profile上下文管理器包装要分析的代码段
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
        # 训练...
        # print('model.training')
        model.train()
        # print('model.trained')
        train_loss = 0.0

        iter_i = 0
        num_iter = len(dataloader_train)
        for batch_inputs, batch_targets in dataloader_train:
            if (iter_i*batch_size)%100==0:
                print("iter [{}/{}],".format(iter_i*batch_size, num_iter*batch_size))
            iter_i += 1
            batch_inputs, batch_targets = batch_inputs.cuda(device=device_ids[0]), batch_targets.cuda(device=device_ids[0])  # 0multi

            # 前向传播...
            ref_output, output_tensor = model.forward(batch_inputs, batch_inputs)

            # 计算损失...
            output_tensor_mean = torch.mean(torch.mean(output_tensor, dim=2, keepdim=True), dim=3, keepdim=True)
            loss = angular_loss(output_tensor_mean, batch_targets)
            train_loss += loss.item()

            # 反向传播...
            loss.backward()

            # 更新模型参数...
            optimizer.step()

            # 清零梯度...
            optimizer.zero_grad()

    print_angular_loss(output_tensor, batch_targets)
    # 将分析结果写入TensorBoard日志文件
    writer.add_profiler("Model Profiler", prof)



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
                val_inputs, val_targets = val_inputs.cuda(device=device_ids[0]), val_targets.cuda(
                    device=device_ids[0])  # 0multi
                val_output = model.forward(val_inputs)
                val_loss += angular_loss(val_output, val_targets).item()

        train_loss /= len(dataloader_train)
        val_loss /= len(dataloader_val)
        print("Epoch [{}/{}], Train Loss: {:.4f}, Validation Loss: {:.4f}"
              .format(epoch + 1, max_epochs, train_loss, val_loss))

        torch.save(model.state_dict(), model_path + 'epoch_' + str(epoch + 1).zfill(4) + '.pth')
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_path + 'best.pth')
            print("Saved the best model at epoch", epoch + 1)
    else:
        # 验证...
        train_loss /= len(dataloader_train)
        print("Epoch [{}/{}], Train Loss: {:.4f},"
              .format(epoch + 1, max_epochs, train_loss))

# 关闭SummaryWriter
writer.close()
