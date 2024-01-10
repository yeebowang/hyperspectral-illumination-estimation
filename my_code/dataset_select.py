import numpy as np
import os

def save_target_data(file_path, target_data):

    with open(file_path, 'w') as file:

        for sample_name, gt_channels in target_data:
            file.write(sample_name)
            for channel in gt_channels:
                file.write(' '+channel)

            file.write('\n')


def load_target_data(file_path):
    target_data = []
    with open(file_path, 'r') as file:
        for line in file:
            for i in range(4):
                sample_name, *gt_channels = line.strip().split()
                # gt_channels = list(map(float, gt_channels))
                sample_name = sample_name + '_' + str(i + 1)
                target_data.append((sample_name, gt_channels))
    return target_data

def load_input_data(input_data_folder, sample_name):
    file_path = os.path.join(input_data_folder, sample_name + '.npy')
    npy_file = np.load(file_path)[0,:,:,:] # from npy WHC to cv2 HWC transpose(1,0,2) then to CHW

    return npy_file

def select_gt(input_data_folder, gt_list, threshold):
    cnt_all = 0
    cnt_select = 0
    target_data = []
    for sample_name, gt_channels in gt_list:
        cnt_all += 1
        npy_file = load_input_data(input_data_folder,sample_name)
        true_ratio = 1-len(np.where(npy_file == np.zeros((31,1,1)))[0])/(npy_file.shape[1]*npy_file.shape[2])
        if true_ratio >= threshold:
            cnt_select += 1
            target_data.append((sample_name, gt_channels))
    print('selected {} samples from {} samples'.format(cnt_select, cnt_all))
    return target_data
def remove_gt(input_data_folder, gt_list, threshold):
    cnt_all = 0
    cnt_select = 0
    target_data = []
    for sample_name, gt_channels in gt_list:
        cnt_all += 1
        npy_file = load_input_data(input_data_folder,sample_name)
        false_ratio = len(np.where(npy_file == np.zeros((31,1,1)))[0])/(npy_file.shape[1]*npy_file.shape[2])
        if false_ratio >= threshold:
            cnt_select += 1
            target_data.append((sample_name, gt_channels))
    print('removed {} samples from {} samples'.format(cnt_select, cnt_all))
    return target_data
# demo
# gt_list = load_target_data('hsi_txt/set_0.txt')
# gt_list_selected = select_gt('hsi_256', gt_list, 1)
# save_target_data('hsi_txt/set_0_256_selected.txt', gt_list_selected)
# select
# gt_list = load_target_data('/home/ybwang/datasets/KAUST-SRI/hsi_txt/set_2.txt')
# gt_list_selected = select_gt('/home/ybwang/datasets/KAUST-SRI/hsi_256', gt_list, 1)
# save_target_data('/home/ybwang/datasets/KAUST-SRI/hsi_txt/set_2_256_selected.txt', gt_list_selected)
# remove
gt_list = load_target_data('/home/ybwang/datasets/KAUST-SRI/hsi_txt/set_2.txt')
gt_list_selected = remove_gt('/home/ybwang/datasets/KAUST-SRI/hsi_256', gt_list, 1)
save_target_data('/home/ybwang/datasets/KAUST-SRI/hsi_txt/set_2_256_remove.txt', gt_list_selected)