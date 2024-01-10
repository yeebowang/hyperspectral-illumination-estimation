import torch
import torch.nn as nn
from data_loader_miid_selected import load_data
from einops import rearrange
from einops.layers.torch import Rearrange
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import math
from collections import OrderedDict
import numpy as np
import scipy.io as sio
import random
import cv2
import torch.nn.functional as F
from argparse import ArgumentParser
parser = ArgumentParser(description='CMFNet')
parser.add_argument('--set', type=int, default=0, help='choose set id: 0,1,2')
args = parser.parse_args()
# 设置超参数
device_ids = [3]  # 可用GPU
batch_size = 16 # 8
num_workers = 8
channel = 31

class LRMF():
    def __init__(self, channel, k=1):
        super(LRMF, self).__init__()
        self.channel = channel
        self.k = k
    def kmeans(self, data):
        flattened_data = data.view(data.size(0), data.size(1), -1)
        # 设置簇的数量 k
        k = self.k

        # 随机初始化 k 个聚类中心，并将聚类中心移动到 GPU 上
        centroids = flattened_data[:, random.sample(range(flattened_data.size(1)), k), :].cuda(device=device_ids[0])

        # 定义最大迭代次数和收敛阈值
        max_iterations = 100
        convergence_threshold = 1e-4

        for iteration in range(max_iterations):
            # 计算每个数据点到所有聚类中心的距离
            distances = torch.norm(flattened_data[:, None] - centroids, dim=2)

            # 获取每个数据点所属的簇索引
            cluster_assignment = torch.argmin(distances, dim=1)

            # 更新聚类中心为每个簇的数据点的平均值
            old_centroids = centroids.clone()
            for i in range(k):
                cluster_points = flattened_data[cluster_assignment == i]
                if len(cluster_points) > 0:
                    centroids[:, i, :] = cluster_points.mean(dim=0)

            # 检查聚类中心是否收敛
            if torch.norm(centroids - old_centroids) < convergence_threshold:
                break

        # 将聚类中心还原成原来的形状 (N, C, H, W)
        centroids = centroids.view(data.size(0), k, -1)

        # 找到属于每个类的样本点
        clusters = [[] for _ in range(k)]
        for i in range(data.size(0)):
            cluster_index = cluster_assignment[i].item()
            clusters[cluster_index].append(data[i])


        for i, cluster in enumerate(clusters):
            print(f"属于聚类中心 {i} 的样本：")
            print(torch.stack(cluster))


    def iterate(self, I, L):
        I_ori = I
        # print('k=', self.k)
        L_1 = L.detach()
        L_2 = L.detach()
        L_tmp = L_1.view(L.size(0), L.size(1), -1).unsqueeze(2)
        L_T = L_2.view(L.size(0), L.size(1), -1).unsqueeze(1)
        # print(L_tmp.shape, L_T.shape)# torch.Size([8, 31, 1, 65536]) torch.Size([8, 1, 31, 65536])
        mul_result = L_tmp * L_T
        mul_result=mul_result.cuda(device=device_ids[0])
        # print('mul_result', mul_result[0,:,:,19909])
        # print(mul_result.shape) # torch.Size([8, 31, 31, 65536])
        mul_result = mul_result.view(L.size(0), -1, L.size(2)*L.size(3))
        mul_result = mul_result.view(L.size(0), -1, L.size(2), L.size(3))
        # print(mul_result.shape) #torch.Size([8, 961, 256, 256])
        E = torch.eye(self.channel).cuda(device=device_ids[0])
        # print('E', E.shape)#E torch.Size([31, 31])
        E = E.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        E = E.expand((L.size(0), L.size(1), L.size(1), L.size(2), L.size(3)))
        E = E.view(L.size(0), -1, L.size(2), L.size(3))
        # print('E',E.shape)#E torch.Size([8, 961, 256, 256])
        mul_result = mul_result / torch.norm(L, 2, dim=1, keepdim=True)
        # print('mul_result_norm', mul_result[0, :, 128, 128])
        # print('eye', E[0, :, 128, 128])
        P = E - mul_result # torch.Size([8, 961, 256, 256])
        P = P.view(L.size(0), L.size(1), L.size(1), L.size(2), L.size(3))# torch.Size([8, 31,31, 256, 256])
        P = P.view(L.size(0), L.size(1), L.size(1), L.size(2)*L.size(3))# torch.Size([8, 31,31, 256*256])

        P = P.transpose(2,3).transpose(1,2)
        # print('P',P.shape)#P torch.Size([8, 65536, 31, 31])
        P = P.reshape(-1, L.size(1), L.size(1))
        # print('P',P[199909, :, :])
        # print('P', P.shape)#P torch.Size([524288, 31, 31])
        I = I.view(L.size(0), L.size(1), -1).transpose(1,2)
        # print('I',I.shape)#I torch.Size([8, 65536, 31])
        I = I.reshape(-1, L.size(1)).unsqueeze(-1)
        # print('I', I[199909, :, :])
        # print('I', I.shape)#I torch.Size([524288, 31, 1])
        PI = torch.bmm(P, I)
        # print('PI', PI[199909, :, :])
        # print('PI',PI.shape)#PI torch.Size([524288, 31, 1])
        PI = PI.reshape(L.size(0),L.size(2)*L.size(3),-1).transpose(1,2).view(L.size(0), L.size(1),L.size(2),L.size(3))
        # print('PI', PI.shape)#PI torch.Size([8, 31, 256,256])
        # clusters = self.kmeans(PI)
        input_tensor = (PI).view(L.size(0), L.size(1), -1)
        I_flat = I_ori.view(L.size(0), L.size(1), -1)
        # print(input_tensor[0,:,19909])
        input_tensor=input_tensor.cuda(device=device_ids[0])
        I_flat=I_flat.cuda(device=device_ids[0])
        # 计算每个像素点对应的光谱的 L2 范数，即在通道维度（第1个维度）上求范数
        norms = torch.norm(input_tensor, p=2, dim=1, keepdim=True) # N, 1, HW
        # norms = torch.sum(input_tensor, dim=1, keepdim=True) # N, 1, HW
        PI_loc = PI.clone()
        PI_loc = PI_loc.view(L.size(0), L.size(1), -1)
        # 找到每个 batch 内最亮的像素索引
        specular_pixel_indices = torch.argmax(norms, dim=2, keepdim=True).squeeze(-1).squeeze(-1) # N, 1, 1
        I_s = torch.zeros(L.size(0), L.size(1),1)
        I_s = I_s.cuda(device=device_ids[0])
        for b in range(L.size(0)):
            I_s[b, :, 0]=I_flat[b, :, specular_pixel_indices[b]]
            PI_loc[b, :, specular_pixel_indices[b]-5:specular_pixel_indices[b]+5] = 1.0
        I_s = I_s.unsqueeze(-1)
        PI_loc = PI_loc.view(L.size(0), L.size(1), L.size(2), L.size(3))
        # I_s = torch.tensor(I_s).to(torch.float32).cuda(device=device_ids[0])
        # print(specular_pixel_indices.shape)
        # print(specular_pixel_indices)
        # # 将索引展平为 (8, 1, 1) 的形状，以便在 input_tensor 中进行 gather 操作
        # specular_pixel_indices = specular_pixel_indices.view(L.size(0), 1, 1)
        # new_index = specular_pixel_indices.expand(L.size(0), L.size(1), -1)
        # print(new_index)
        # for c in range(L.size(1)):
        #     new_index[:, c, :] = c
        # print(new_index)
        # # 从 input_tensor 中收集最亮像素对应的光谱
        # specular_pixels_spectra = torch.gather(input_tensor, dim=2, index=new_index)
        # I_s = specular_pixels_spectra
        # I_s = I_s.unsqueeze(-1)
        # I_s = I_s.cuda(device=device_ids[0])
        # print(I_s.shape)
        # print(I_s.detach().cpu().numpy()[0,:,0,0])
        # I_d = I - I_s

        return I_s, PI, PI_loc
    def draw(self, I, PI, PI_loc, names):
        I = I.detach().cpu().numpy()
        PI = PI.detach().cpu().numpy()
        PI_loc = PI_loc.detach().cpu().numpy()

        for i in range(PI.shape[0]):
            name = names[i]
            img_1 = PI[i, [channel*1//6, channel*3//6, channel*5//6], :, :]
            img_1 = img_1.transpose(1,2,0)
            img_1 = img_1 / img_1.max() * 255
            img_2 = I[i, [channel*1//6, channel*3//6, channel*5//6], :, :]
            img_2 = img_2.transpose(1, 2, 0)
            img_2 = img_2 / img_2.max() * 255
            img_3 = PI_loc[i, [channel*1//6, channel*3//6, channel*5//6], :, :]
            img_3 = img_3.transpose(1,2,0)
            img_3 = img_3 / img_3.max() * 255
            cv2.imwrite('PI_png/'+str(i).zfill(2) + name + '_PI.png', img_1)
            cv2.imwrite('PI_png/'+str(i).zfill(2) + name + '_I.png', img_2)
            cv2.imwrite('PI_png/'+str(i).zfill(2) + name + '_PI_loc.png', img_3)

    def test(self, I):
        L = torch.mean(torch.mean(I, dim=2, keepdim=True), dim=3, keepdim=True)
        L = L.expand(I.shape)
        out_L, PI, PI_loc = self.iterate(I, L)
        out_L = out_L.expand(I.shape)

        # for i in range(4):
        #     out_L = self.iterate(I, out_L)
        #     out_L = out_L.expand(I.shape)
        return out_L, PI, PI_loc

# 计算角误差损失函数
RAD2DEG = 180. / math.pi
def print_angular_loss(pred, target):
    pred = pred[0,:]
    target = target[0,:]
    pred_norm = pred / (torch.norm(pred)+1e-9)
    target_norm = target / (torch.norm(target)+1e-9)
    print('pred_norm', pred_norm)
    print('target_norm',target_norm)
def absolute_loss(pred, target):

    numerator = torch.sum(pred * target, dim=1)  # shape: (N,)
    denominator = torch.sum(pred * pred, dim=1)  # shape: (N,)
    # 计算误差
    error = torch.norm(target - pred * numerator.unsqueeze(1) / denominator.unsqueeze(1), dim=1,
                       p=1)  # shape: (N,)
    return error.mean()
    # return np.linalg.norm(target - pred*np.dot(pred, target)/np.dot(pred, pred), ord=1)

def angular_loss(pred, target):#35.1754

    # 归一化预测和目标
    pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True)+1e-9)
    target_norm = target / (torch.norm(target, dim=1, keepdim=True)+1e-9)
    # 计算角误差
    loss = torch.acos(torch.clamp(torch.sum(pred_norm * target_norm, dim=1), min=-1.0, max=1.0))
    return loss.mean()* RAD2DEG


# res_dir = '../prec_illum/Gray_Edge_norm_1_sigma_1/'
res_dir = '../prec_illum/LRMF_v0/'# v0 find PI MAX norm, v0.1 find PI-L, MAX v0.2 find PI MAX sum, V0 IS BEST
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
# 数据加载器
dataloader_train, dataloader_test = load_data(batch_size, num_workers, args.set)
model = LRMF(channel=channel, k=1)
# 测试循环

with torch.no_grad():
    angular_error = 0.0
    absolute_error = 0.0
    batch_i = 0
    for test_inputs, test_targets, _, sample_names in dataloader_test:
        batch_i+=1
        print("batch[{}/{}]".format(batch_i, len(dataloader_test)))
        test_inputs, test_targets = test_inputs.cuda(device=device_ids[0]), test_targets.cuda(device=device_ids[0])


        test_output, PI, PI_loc = model.test(test_inputs)
        model.draw(test_inputs, PI, PI_loc, sample_names)


        # save mat
        for i in range(len(test_output)):
            pred = test_output[i]
            # print(pred)
            prec_light = pred.detach().cpu().numpy()
            prec_light = prec_light[np.newaxis, :31, :, :]
            # print(prec_light)
            # prec_light = np.tile(prec_light, (1, 1, 256, 256))
            print(prec_light.shape) # check for 1,31,256,256
            name = sample_names[i]
            sio.savemat(res_dir + 'res_' + name + '.mat', {'predict_light': prec_light})

        test_output = torch.mean(torch.mean(test_output, dim=2, keepdim=True), dim=3, keepdim=True)
        test_targets = torch.mean(torch.mean(test_targets, dim=2, keepdim=True), dim=3, keepdim=True) # for miid dataloader
        test_output = test_output.squeeze(dim=-1).squeeze(dim=-1)[:, :31]
        test_targets = test_targets.squeeze(dim=-1).squeeze(dim=-1)[:, :31]

        print_angular_loss(test_output, test_targets)
        angular_error += angular_loss(test_output, test_targets).item()
        absolute_error += absolute_loss(test_output, test_targets).item()

angular_error /= len(dataloader_test)
absolute_error /= len(dataloader_test)
print("Angular Error: {:.4f}  Absolute Error: {:.4f}".format(angular_error, absolute_error))
