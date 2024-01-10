import torch
import torch.nn as nn
import numpy as np
import random
from data_loader_256 import load_data
import os
import math
from model.cmf_net import CMFNET,count_parameters
import scipy.io as sio
# 设置超参数...

device = torch.device("cuda")

device_ids = [0, 1, 2, 3]  # 可用GPU
# device_ids = [0]  # 可用GPU/
batch_size = 2 * len(device_ids)
# batch_size = 1
num_workers = 4
channel = 31

max_epochs = 3000
learning_rate = 1e-3 # 3e-4
save_interval = 30
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


model = CMFNET()
# 计算角误差损失函数...
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


# 指定要用到的设备...
model = torch.nn.DataParallel(model, device_ids=device_ids)  # 0multi
model = model.cuda(device=device_ids[0])  # 0multi
# 加载模型权重
model.load_state_dict(torch.load(model_path + 'epoch_0150.pth'))
res_dir = '../prec_illum/MIID_multi/epoch_0150/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


dataloader_train, dataloader_test = load_data(batch_size, num_workers)

# 测试循环
model.eval()
with torch.no_grad():
    angular_error = 0.0
    absolute_error = 0.0
    for test_inputs, test_targets, test_names in dataloader_test:
        test_inputs, test_targets = test_inputs.cuda(device=device_ids[0]), test_targets.cuda(device=device_ids[0])
        test_output = model.forward(test_inputs)
        # save mat
        for i in range(batch_size):
            pred = test_output[i]
            prec_light = pred.detach().cpu().numpy()
            prec_light = prec_light[np.newaxis,:,:,:]
            print(prec_light.shape)
            name = test_names[i]

            sio.savemat(res_dir + 'res_' + name + '.mat', {'predict_light': prec_light})

        # end save
        test_output = torch.mean(torch.mean(test_output, dim=2, keepdim=True), dim=3, keepdim=True)
        angular_error += angular_loss(test_output, test_targets).item()
        absolute_error += absolute_loss(test_output, test_targets).item()

angular_error /= len(dataloader_test)
absolute_error /= len(dataloader_test)
print("Angular Error: {:.4f}  Absolute Error: {:.4f}".format(angular_error, absolute_error))
