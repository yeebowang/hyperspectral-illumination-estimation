import torch
import torch.nn as nn
from data_loader_hsisi import load_data
from einops import rearrange
from einops.layers.torch import Rearrange
import os
import math
from collections import OrderedDict
import numpy as np
import scipy.io as sio
from argparse import ArgumentParser
parser = ArgumentParser(description='CMFNet')
parser.add_argument('--set', type=int, default=0, help='choose set id: 0,1,2')
args = parser.parse_args()
# 设置超参数
device_ids = [2,3]  # 可用GPU
batch_size = 8 * len(device_ids)
num_workers = 4
channel = 34
# height = 512
# width = 512
image_size = 512
patch_size = 8
num_classes = channel
dim = 34  # 256
depth = 6
heads = 2  # 8
mlp_dim = 512
model_path = '../model_output/ViT_2_s'+str(args.set)+'/'
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

class SpectrumEstimationViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, channel, dim, depth, heads, mlp_dim):
        super(SpectrumEstimationViT, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channel  # Assuming input is a -channel image

        self.patch_embedding = nn.Conv2d(patch_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim), num_layers=depth)

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

        return x

# 创建模型实例
model = SpectrumEstimationViT(image_size, patch_size, num_classes, channel, dim, depth, heads, mlp_dim)

# 指定要用到的设备
model = torch.nn.DataParallel(model, device_ids=device_ids)
# 模型加载到设备0
model = model.cuda(device=device_ids[0])

# 加载模型权重
model.load_state_dict(torch.load(model_path + 'epoch_3000.pth'))
res_dir = '../prec_illum/ViT_2/HSISI/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
# 数据加载器
dataloader_train, dataloader_test = load_data(batch_size, num_workers, args.set)

# 测试循环
model.eval()
with torch.no_grad():
    angular_error = 0.0
    absolute_error = 0.0
    for test_inputs, test_targets, sample_names in dataloader_test:
        test_inputs, test_targets = test_inputs.cuda(device=device_ids[0]), test_targets.cuda(device=device_ids[0])
        test_output = model.forward(test_inputs)

        # save mat
        for i in range(len(sample_names)):
            pred = test_output[i]
            prec_light = pred.detach().cpu().numpy()
            prec_light = prec_light[np.newaxis, :31, np.newaxis, np.newaxis]
            prec_light = np.tile(prec_light, (1, 1, 256, 256))
            print(prec_light.shape)
            name = sample_names[i]
            sio.savemat(res_dir + 'res_' + name + '.mat', {'predict_light': prec_light})

        angular_error += angular_loss(test_output, test_targets).item()
        absolute_error += absolute_loss(test_output, test_targets).item()

angular_error /= len(dataloader_test)
absolute_error /= len(dataloader_test)
print("Angular Error: {:.4f}  Absolute Error: {:.4f}".format(angular_error, absolute_error))
