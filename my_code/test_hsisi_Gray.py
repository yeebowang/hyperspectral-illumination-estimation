import torch
import torch.nn as nn
from data_loader_hsisi_2 import load_data
from einops import rearrange
from einops.layers.torch import Rearrange
import os
import math
from collections import OrderedDict
import numpy as np
import scipy.io as sio
import torch.nn.functional as F
from argparse import ArgumentParser
parser = ArgumentParser(description='CMFNet')
parser.add_argument('--set', type=int, default=0, help='choose set id: 0,1,2')
args = parser.parse_args()
# 设置超参数
device_ids = [0,1,2,3]  # 可用GPU
batch_size = 32
num_workers = 8
def gray_edge(image, order=1, p=2, sigma=1.0):
    """
    Gray Edge algorithm with Gaussian smoothing for color correction on multispectral images.

    Parameters:
        image (torch.Tensor): Input multispectral image tensor (shape: [batch_size, channels, height, width]).
        order (int): Order of Gray Edge. 0 for Gray World, 1 for 1st Order Gray Edge, 2 for 2nd Order Gray Edge.
        sigma (float): Standard deviation of Gaussian kernel for smoothing.
        p (int): Norm type for calculating edge magnitude (default: 2).

    Returns:
        torch.Tensor: Color corrected multispectral image tensor (shape: [batch_size, channels, height, width]).
    """
    assert order in [0, 1, 2], "Invalid order. Use 0 for Gray World, 1 for 1st Order Gray Edge, 2 for 2nd Order Gray Edge."


    if sigma > 0:
        # Apply Gaussian smoothing to the input image
        image = gaussian_blur(image, sigma)

    if order == 0:
        # Gray World algorithm without gradients and smoothing
        mean_color = torch.norm(torch.norm(image, p=p, dim=2, keepdim=True), p=p, dim=3, keepdim=True)
        illum = mean_color / torch.mean(mean_color, dim=1, keepdim=True)
        print(illum.shape)

    else:

        # Compute gradients
        if order == 1:
            # 1st Order Gray Edge using Sobel operator
            gradient_x, gradient_y = sobel_gradients(image)
            gradient = torch.sqrt(gradient_x**2+gradient_y**2)
            # print('grad',gradient_x.shape)
        if order == 2:
            # 2nd Order Gray Edge using Laplacian operator
            gradient = laplacian_gradients(image)
        # print('grad',gradient)
        # print('grad.shape',gradient.shape)
        # Calculate edge magnitude
        edge_magnitude = torch.norm(torch.norm(gradient, p=p, dim=2, keepdim=True), p=p, dim=3, keepdim=True)
        # print('edge_mag',edge_magnitude)
        illum = edge_magnitude / torch.mean(edge_magnitude, dim=1, keepdim=True)
        print(illum.shape)

    return illum

def gaussian_kernel(kernel_size, sigma):
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    gaussian_kernel = (1./(2.*np.pi*variance)) *\
                      torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) /\
                                (2*variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    return gaussian_kernel

def create_gaussian_kernel(sigma, num_channels):
    kernel_size = int(6*sigma) + 1
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    # kernel = kernel.expand(num_channels, num_channels, kernel_size, kernel_size)
    return kernel

def gaussian_blur(image, sigma):
    batch_size, num_channels, height, width = image.shape
    kernel = create_gaussian_kernel(sigma, num_channels).to(image.device)
    image_groups = torch.split(image, 1, dim=1)
    output_groups = []
    for group in image_groups:
        # 在这里进行每个组的计算
        output_i = F.conv2d(group, kernel, padding=kernel.shape[-1]//2)
        output_groups.append(output_i)

    # 将结果组合成一个张量，沿着dim=1维度连接回去
    output = torch.cat(output_groups, dim=1)

    return output

def sobel_gradients(image):
    # Define the Sobel operators for x and y directions
    sobel_x_kernel = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
    # print('kernel',sobel_x_kernel.shape)
    # print(sobel_x_kernel)
    sobel_y_kernel = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)
    image_groups = torch.split(image, 1, dim=1)
    sobel_x_groups = []
    sobel_y_groups = []
    for group in image_groups:
        # 在这里进行每个组的计算
        # Apply Sobel operators to each channel separately
        sobel_x = F.conv2d(group, sobel_x_kernel.to(image.device), padding=1)
        sobel_y = F.conv2d(group, sobel_y_kernel.to(image.device), padding=1)
        sobel_x_groups.append(sobel_x)
        sobel_y_groups.append(sobel_y)

    # 将结果组合成一个张量，沿着dim=1维度连接回去
    output_x = torch.cat(sobel_x_groups, dim=1)
    output_y = torch.cat(sobel_y_groups, dim=1)

    return output_x, output_y


def laplacian_gradients(image):
    # Define the Laplacian operator
    laplacian_kernel = torch.FloatTensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]]).to(image.device)

    # Apply Laplacian operator to each channel separately
    # gradient = F.conv2d(image, laplacian_kernel, padding=1, groups=image.shape[1])
    # gradient = F.conv2d(image, laplacian_kernel, padding=1)
    # gradient_x = gradient[:, :, :, 1:-1]  # Remove the first and last columns (padding)
    # gradient_y = gradient[:, :, 1:-1, :]  # Remove the first and last rows (padding)
    # return gradient
    image_groups = torch.split(image, 1, dim=1)
    output_groups = []
    for group in image_groups:
        # 在这里进行每个组的计算
        output_i = F.conv2d(group, laplacian_kernel, padding=1)
        output_groups.append(output_i)

    # 将结果组合成一个张量，沿着dim=1维度连接回去
    output = torch.cat(output_groups, dim=1)

    return output
# 示例用法
# 假设image是一个形状为[batch_size, channels, height, width]的多光谱图像张量，将张量移动到GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 计算角误差损失函数
RAD2DEG = 180. / math.pi
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


# res_dir = '../prec_illum/Gray_Edge_norm_inf_sigma_1/'
res_dir = '../prec_illum/Gray_World/HSISI/'
# res_dir = '../prec_illum/Max_RGB/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
# 数据加载器
dataloader_train, dataloader_test = load_data(batch_size, num_workers, args.set)

# 测试循环

with torch.no_grad():
    angular_error = 0.0
    absolute_error = 0.0
    for test_inputs, test_targets, _, sample_names in dataloader_test:
        test_inputs, test_targets = test_inputs.cuda(device=device_ids[0]), test_targets.cuda(device=device_ids[0])
        test_output = gray_edge(test_inputs, order=0, p=1, sigma=0) # Gray World
        # test_output = gray_edge(test_inputs, order=0, p=float('inf'), sigma=0) # Max
        # test_output = gray_edge(test_inputs, order=0, p=float('inf'), sigma=1.0) # Gray Edge
        # test_output = gray_edge(test_inputs, order=1, p=float('inf'), sigma=1.0) # 1st Order Gray Edge
        # test_output = gray_edge(test_inputs, order=2, p=float('inf'), sigma=0) # 2nd Order Gray Edge


        # save mat
        for i in range(len(test_output)):
            pred = test_output[i]
            # print(pred)
            prec_light = pred.detach().cpu().numpy()
            prec_light = prec_light[np.newaxis, :31, :, :]
            prec_light = np.tile(prec_light, (1, 1, 256, 256))
            print(prec_light.shape) # check for 1,31,256,256
            name = sample_names[i]
            sio.savemat(res_dir + 'res_' + name + '.mat', {'predict_light': prec_light})

        test_output = torch.mean(torch.mean(test_output, dim=2, keepdim=True), dim=3, keepdim=True)
        test_targets = torch.mean(torch.mean(test_targets, dim=2, keepdim=True), dim=3, keepdim=True) # for miid dataloader
        test_output = test_output.squeeze(dim=-1).squeeze(dim=-1)[:, :31]
        test_targets = test_targets.squeeze(dim=-1).squeeze(dim=-1)[:, :31]


        angular_error += angular_loss(test_output, test_targets).item()
        absolute_error += absolute_loss(test_output, test_targets).item()

angular_error /= len(dataloader_test)
absolute_error /= len(dataloader_test)
print("Angular Error: {:.4f}  Absolute Error: {:.4f}".format(angular_error, absolute_error))
