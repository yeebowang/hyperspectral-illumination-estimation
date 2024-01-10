import torch
import torch.nn as nn
import numpy as np
import random
from data_loader import load_data
from einops import rearrange
from einops.layers.torch import Rearrange
import os
import math
import torch.nn.functional as F
from collections import OrderedDict
# 设置超参数
device_ids = [0, 1,2,3] # 可用GPU
batch_size = 8 * len(device_ids)
num_workers = 8
channel = 34
height = 512
width = 512
image_size = 512
patch_size = 32
num_classes = channel
dim = 34 # 256
depth = 6
heads = 2 # 8
mlp_dim = 512
max_epochs = 1000
save_interval = 10
val_interval = 10
best_loss = 1e6
model_path = '../model_output/ViT_hd2_ps32/'
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



class SpectrumEstimationViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, channel, dim, depth, heads, mlp_dim):
        super(SpectrumEstimationViT, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channel  # Assuming input is a -channel image

        self.patch_embedding = nn.Conv2d(patch_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim), num_layers=depth)
        # self.fc = nn.Linear(dim, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
    def forward(self, x):
        x = self.patch_embedding(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        b, n, _ = x.shape

        # Add positional embedding
        x = torch.cat((x, self.positional_embedding.repeat(b, 1, 1)), dim=1)

        # Transformer encoding
        x = self.transformer(x)

        # Classification
        x = x[:, 0, :]  # Take the first token representation
        # x = self.fc(x)

        return x


# 计算角误差损失函数

def absolute_loss(pred, target):
    pred = pred[:,:31]
    target = target[:,:31]
    # 归一化预测和目标
    # pred = pred / (torch.norm(pred, dim=1, keepdim=True)+1e-9)
    # target = target / (torch.norm(target, dim=1, keepdim=True)+1e-9)

    numerator = torch.sum(pred * target, dim=1)  # shape: (N,)
    denominator = torch.sum(pred * pred, dim=1)  # shape: (N,)
    s = numerator / denominator

    # 计算误差
    error = torch.norm(target - pred * s.unsqueeze(1), dim=1,
                       p=1)  # shape: (N,)

    return error.mean()
RAD2DEG = 180. / math.pi
def angular_loss(pred, target):#35.1754
    # print('pred', pred.shape)
    # print('target',target.shape)
    pred = pred[:,:31]
    target = target[:,:31]

    # 归一化预测和目标
    pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True)+1e-9)
    target_norm = target / (torch.norm(target, dim=1, keepdim=True)+1e-9)
    # print('pred_norm', pred_norm.shape)
    # print('target_norm',target_norm.shape)
    # 计算角误差
    loss = torch.acos(torch.clamp(torch.sum(pred_norm * target_norm, dim=1), min=-1.0, max=1.0))
    return loss.mean()* RAD2DEG
def print_angular_loss(pred, target):#35.1754
    pred = pred[:,:31]
    target = target[:,:31]
    # 归一化预测和目标
    pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True)+1e-9)
    target_norm = target / (torch.norm(target, dim=1, keepdim=True)+1e-9)

    print('pred_norm', pred_norm)
    print('target_norm',target_norm)


# 创建模型实例
model = SpectrumEstimationViT(image_size, patch_size, num_classes, channel, dim, depth, heads, mlp_dim)

# 指定要用到的设备
model = torch.nn.DataParallel(model, device_ids=device_ids)
# 模型加载到设备0
model = model.cuda(device=device_ids[0])

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-4)
dataloader_train, dataloader_test = load_data(batch_size, num_workers)
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
        loss = angular_loss(output_tensor, batch_targets)
        train_loss_ae += loss.item()
        train_loss_abe += absolute_loss(output_tensor, batch_targets).item()
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
                val_loss_ae += angular_loss(val_output, val_targets).item()
                val_loss_abe += absolute_loss(val_output, val_targets).item()

        print_angular_loss(val_output, val_targets)
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
