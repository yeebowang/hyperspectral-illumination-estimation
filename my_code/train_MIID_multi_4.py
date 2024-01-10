import torch
import torch.nn as nn
import numpy as np
import random
from data_loader_miid import load_data
import os
import math
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.models as models
from model.cmf_net_3 import CMFNET,get_Initial



# 设置超参数...
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda")

device_ids = [0,1,2,3]  # 可用GPU
# device_ids = [0]  # 可用GPU/
batch_size = 2 * len(device_ids)
# batch_size = 1
num_workers = 8
channel = 31

max_epochs = 300
start_epoch = 0
learning_rate = 3e-4 # 3e-4
save_interval = 5
best_loss = 1e6
model_path = '../model_output/MIID_multi_4_batch_8/'
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
def absolute_loss(pred, target):
    pred = pred.squeeze(dim=-1).squeeze(dim=-1)[:,:31]
    target = target.squeeze(dim=-1).squeeze(dim=-1)[:,:31]
    # 归一化预测和目标
    pred = pred / (torch.norm(pred, dim=1, keepdim=True)+1e-9)
    target = target / (torch.norm(target, dim=1, keepdim=True)+1e-9)

    numerator = torch.sum(pred * target, dim=1)  # shape: (N,)
    denominator = torch.sum(pred * pred, dim=1)  # shape: (N,)
    s = numerator / denominator

    # 计算误差
    error = torch.norm(target - pred * s.unsqueeze(1), dim=1,
                       p=1)  # shape: (N,)

    return error.mean()

def mean_square_loss(pred_illum, target_illum, pred_respo, target_respo):
    pred_respo = pred_respo[:, :31, :, :]
    target_respo = target_respo[:, :31, :, :]
    pred_illum = pred_illum[:,:31,:,:]
    target_illum = target_illum[:,:31,:,:]

    numerator = torch.sum(pred_illum * target_illum, dim=(1,2,3))  # shape: (N,)
    denominator = torch.sum(pred_illum * pred_illum, dim=(1,2,3))  # shape: (N,)
    s = numerator/denominator
    s_4D = s.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    mse_L = torch.norm(s_4D*pred_illum-target_illum,p=2, dim=1)
    mse_R = torch.norm(pred_respo/s_4D-target_respo,p=2, dim=1)
    mse_L = torch.norm(mse_L, p=2, dim=-1)
    mse_R = torch.norm(mse_R, p=2, dim=-1)
    mse_L = torch.norm(mse_L, p=2, dim=-1)
    mse_R = torch.norm(mse_R, p=2, dim=-1)

    return (mse_L + 0.2*mse_R + 0.001*torch.abs(torch.log(s.unsqueeze(-1)))).mean()

RAD2DEG = 180. / math.pi
def angular_loss(pred, target):#35.1754

    # max_val, _ = torch.max(pred, dim=1)
    # min_val, _ = torch.min(pred, dim=1)
    # pred = pred.transpose(1,0)
    # pred = (pred - min_val)/(max_val - min_val)
    # pred = pred.transpose(1, 0)

    pred = pred.squeeze(dim=-1).squeeze(dim=-1)[:,:31]
    target = target.squeeze(dim=-1).squeeze(dim=-1)[:,:31]

    # 归一化预测和目标
    pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True)+1e-9)
    target_norm = target / (torch.norm(target, dim=1, keepdim=True)+1e-9)
    # print('pred_norm', pred_norm)
    # print('target_norm',target_norm)
    # 计算角误差
    loss = torch.acos(torch.clamp(torch.sum(pred_norm * target_norm, dim=1), min=-1.0, max=1.0))
    return loss.mean()* RAD2DEG
def print_angular_loss(pred, target):#35.1754
    pred = pred.squeeze(dim=-1).squeeze(dim=-1)[:,:31]
    target = target.squeeze(dim=-1).squeeze(dim=-1)[:,:31]
    # 归一化预测和目标
    pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True)+1e-9)
    target_norm = target / (torch.norm(target, dim=1, keepdim=True)+1e-9)

    print('pred_norm', pred_norm)
    print('target_norm',target_norm)
    # 计算角误差


# 创建SummaryWriter对象


# 指定要用到的设备...
model = torch.nn.DataParallel(model, device_ids=device_ids)  # 0multi
model = model.cuda(device=device_ids[0])  # 0multi
if start_epoch>0:
    model.load_state_dict(torch.load(model_path + 'epoch_'+str(start_epoch).zfill(4)+'.pth'))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


print('start loading data...')
dataloader_train, dataloader_val = load_data(batch_size, num_workers)
print('start training...')
# 训练循环...
torch.autograd.set_detect_anomaly(False)
for epoch in range(start_epoch,max_epochs):
    # torch.cuda.empty_cache()
    # dataloader_train, dataloader_val = load_data(batch_size, num_workers)
    # model = model.cpu()
    # 训练
    model.train()
    # model.eval()
    train_loss_mse = 0.0
    train_loss_ae = 0.0
    train_loss_abe = 0.0

    # 清零梯度
    # optimizer.zero_grad()

    iter_i = 0
    num_iter = len(dataloader_train)
    # with torch.no_grad():
    if 1:
        for batch_inputs, batch_gt_L, batch_gt_R, batch_names in dataloader_train:
            torch.cuda.empty_cache()
            # max_grad_norm = 1.0  # 设置梯度裁剪的最大范数
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            # torch.autograd.set_grad_enabled(True)
            if (iter_i*batch_size)%100==0:
                print("iter [{}/{}],".format(iter_i*batch_size, num_iter*batch_size))
                # print(count_parameters(model))
            iter_i += 1
            batch_inital = get_Initial(batch_inputs, 1, 'max')
            batch_inital, batch_gt_L, batch_gt_R = batch_inital.cuda(device=device_ids[0]), \
                batch_gt_L.cuda(device=device_ids[0]), batch_gt_R.cuda(device=device_ids[0])  # 0multi

            # 前向传播...
            # batch_output_L, batch_output_R = model.forward(batch_gt_L, batch_inital)
            batch_output_L, batch_output_R =  model.forward(batch_inital, batch_inital)

            # 计算损失...
            output_L_mean = torch.mean(torch.mean(batch_output_L, dim=2, keepdim=True), dim=3, keepdim=True)
            batch_gt_L_mean = torch.mean(torch.mean(batch_gt_L, dim=2, keepdim=True), dim=3, keepdim=True)
            loss_ae = angular_loss(output_L_mean, batch_gt_L_mean)
            loss_abe = absolute_loss(output_L_mean, batch_gt_L_mean)

            loss = mean_square_loss(batch_output_L, batch_gt_L, batch_output_R, batch_gt_R)
            train_loss_mse += loss.item()
            train_loss_ae += loss_ae.item()
            train_loss_abe += loss_abe.item()
            # 清零梯度...
            optimizer.zero_grad()
            # 反向传播...
            loss.backward()
            # loss_ae.backward()
            # print('line161')
            # 更新模型参数...
            optimizer.step()
            # torch.autograd.set_grad_enabled(False)
            # print('line164')

            if iter_i==num_iter:
                print_angular_loss(output_L_mean, batch_gt_L_mean)
            # del loss
            # del output_tensor
            # del output_L_mean
            # del batch_inputs
            # del batch_gt_L
            # torch.cuda.empty_cache() # this is the key code


    # 保存模型...
    if (epoch + 1) % save_interval == 0 or epoch == 0:
        # 验证...
        model.eval()
        val_loss_mse = 0.0
        val_loss_ae = 0.0
        val_loss_abe = 0.0

        iter_i = 0
        num_iter = len(dataloader_val)
        with torch.no_grad():

            for batch_inputs, batch_gt_L, batch_gt_R, batch_names in dataloader_val:
                torch.cuda.empty_cache()
                if (iter_i * batch_size) % 100 == 0:
                    print("iter [{}/{}],".format(iter_i * batch_size, num_iter * batch_size))
                    # print(count_parameters(model))
                iter_i += 1
                batch_inital = get_Initial(batch_inputs, 1, 'max')
                batch_gt_R = get_Initial(batch_gt_R, 1, 'max')

                batch_inital, batch_gt_L, batch_gt_R = batch_inital.cuda(device=device_ids[0]), \
                    batch_gt_L.cuda(device=device_ids[0]), batch_gt_R.cuda(device=device_ids[0])  # 0multi

                # 前向传播...
                batch_output_L, batch_output_R = model.forward(batch_inital, batch_inital)

                # 计算损失...
                output_L_mean = torch.mean(torch.mean(batch_output_L, dim=2, keepdim=True), dim=3, keepdim=True)
                batch_gt_L_mean = torch.mean(torch.mean(batch_gt_L, dim=2, keepdim=True), dim=3, keepdim=True)
                loss_ae = angular_loss(output_L_mean, batch_gt_L_mean)
                loss_abe = absolute_loss(output_L_mean, batch_gt_L_mean)

                loss = mean_square_loss(batch_output_L, batch_gt_L, batch_output_R, batch_gt_R)
                val_loss_mse += loss.item()
                val_loss_ae += loss_ae.item()
                val_loss_abe += loss_abe.item()


        train_loss_mse /= len(dataloader_train)
        val_loss_mse /= len(dataloader_val)
        train_loss_ae /= len(dataloader_train)
        train_loss_abe /= len(dataloader_train)
        val_loss_ae /= len(dataloader_val)
        val_loss_abe /= len(dataloader_val)
        print("Epoch [{}/{}], Train ae: {:.4f}, Train abe: {:.4f}, Train mse: {:.4f}, "
              "Val ae: {:.4f}, Val abe: {:.4f}, Val mse: {:.4f}"
              .format(epoch + 1, max_epochs, train_loss_ae, train_loss_abe, train_loss_mse,
                      val_loss_ae, val_loss_abe, val_loss_mse))

        torch.save(model.state_dict(), model_path + 'epoch_' + str(epoch + 1).zfill(4) + '.pth')
        if val_loss_ae < best_loss:
            best_loss = val_loss_ae
            torch.save(model.state_dict(), model_path + 'best.pth')
            print("Saved the best model at epoch", epoch + 1)

    else:
        # 验证...
        train_loss_mse /= len(dataloader_train)
        train_loss_ae /= len(dataloader_train)
        train_loss_abe /= len(dataloader_train)
        print("Epoch [{}/{}], Train ae: {:.4f}, Train abe: {:.4f}, Train mse: {:.4f}"
              .format(epoch + 1, max_epochs, train_loss_ae, train_loss_abe, train_loss_mse))

# 关闭SummaryWriter