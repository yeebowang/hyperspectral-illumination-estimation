import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, target_file, input_folder):

        self.target_data = self.load_target_data(target_file)
        self.input_data_folder = input_folder

    def load_target_data(self, file_path):
        target_data = []
        with open(file_path, 'r') as file:
            for line in file:
                sample_name, *gt_channels = line.strip().split()
                gt_channels = list(map(float, gt_channels))
                target_data.append((sample_name, gt_channels))
        return target_data

    def load_input_data(self, sample_name):
        file_path = os.path.join(self.input_data_folder, sample_name + '.npy')
        npy_file = np.load(file_path).transpose(2, 1, 0) # from npy WHC to cv2 HWC transpose(1,0,2) then to CHW
        npy_file = npy_file/npy_file.max()
        input_tensor = torch.tensor(npy_file)
        input_tensor = input_tensor.to(torch.float32)
        return input_tensor

    def __getitem__(self, index):
        sample_name, gt_channels = self.target_data[index]
        gt_channels = torch.tensor(gt_channels)#.unsqueeze(1).unsqueeze(2)
        gt_channels = gt_channels.to(torch.float32)[:34]
        # print('gt_channels',gt_channels, gt_channels.shape)

        input_data = self.load_input_data(sample_name)[:960, :1056, :34]
        # print('input_data',input_data, input_data.shape)
        return input_data, gt_channels, sample_name

    def __len__(self):
        return len(self.target_data)

def merge_txt_files(file1, file2, merged_file):
    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(merged_file, 'w') as merged_f:
        merged_f.write(f1.read())
        merged_f.write(f2.read())


def load_data(batch_size, num_workers, set):
    # 合并后的文件路径
    if set==0:
        target_file_train = '/home/ybwang/datasets/HSISI/hsi/name_illum.txt'
        target_file_test = '/home/ybwang/datasets/HSISI/hsi/name_illum.txt'

    # 调用函数进行合并
    # merge_txt_files(target_file1, target_file2, target_file_train)


    input_folder = '/home/ybwang/datasets/HSISI/hsi'

    dataset_train = MyDataset(target_file_train, input_folder)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers) # T

    # dataset_train = MyDataset(target_file1, target_file2, target_file_test, input_folder)
    # dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)
    # Example usage of the train dataloader
    # for batch_input, batch_gt in dataloader_train:
        # Training code goes here

    dataset_test = MyDataset(target_file_test, input_folder)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Example usage of the test dataloader
    # for batch_input, batch_gt in dataloader_test:
        # Testing code goes here

    return dataloader_train, dataloader_test