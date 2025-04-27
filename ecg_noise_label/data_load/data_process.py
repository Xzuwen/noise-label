import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter

from .utils import noisify


class MyDataSet_train(Dataset):  # 定义类，用于构建数据集
    def __init__(self, data, clean_label, noise_label):
        self.data = torch.from_numpy(data).float()
        self.num_classes = int(clean_label.max() + 1)
        self.clean_label = torch.from_numpy(clean_label).float()
        self.noise_label = torch.from_numpy(noise_label).float()
        self.length = clean_label.shape[0]

        self.soft_clean_labels = np.zeros((len(self.clean_label), self.num_classes), dtype=np.float32)
        self.soft_noisy_labels = np.zeros((len(self.noise_label), self.num_classes), dtype=np.float32)

        self.soft_clean_labels[np.arange(len(self.clean_label)), self.clean_label.int()] = 1
        self.soft_noisy_labels[np.arange(len(self.noise_label)), self.noise_label.int()] = 1


    def __getitem__(self, index):
        return index, self.data[index], self.clean_label[index], self.noise_label[index]

    def __len__(self):
        return self.length

class MyDataSet(Dataset):  # 定义类，用于构建数据集
    def __init__(self, data, label):
        self.data = torch.from_numpy(data).float()
        self.label = torch.from_numpy(label).float()
        self.length = label.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.length

# =============================================================================
# 加载数据
# =============================================================================
def load_data(path, dataset, batch_size, noise_type, noise_rate, seed):
    '''
    读取数据
    '''
    if dataset == 'afdb':
        x_train = np.load(path + '/afdb/train_data.npy')
        y_train = np.load(path + '/afdb/train_label.npy')
        x_val = np.load(path + '/afdb/test_data.npy')
        y_val = np.load(path + '/afdb/test_label.npy')
    elif dataset == 'adb':
        x_train = np.load(path + '/adb/adb_train_data.npy')
        y_train = np.load(path + '/adb/adb_train_label.npy')
        x_val = np.load(path + '/adb/adb_test_data.npy')
        y_val = np.load(path + '/adb/adb_test_label.npy')
    elif dataset == 'NSA':
        x_train = np.load(path + '/NVA/train_data.npy')
        y_train = np.load(path + '/NVA/train_label.npy')
        x_val = np.load(path + '/NVA/test_data.npy')
        y_val = np.load(path + '/NVA/test_label.npy')

    print('x_train.shape, x_val.shape: ', x_train.shape, x_val.shape)

    # 统计类别数量分布
    unique_classes, counts = np.unique(y_train, return_counts=True)
    for class_label, count in zip(unique_classes, counts):
        print(f"y_train类别 {class_label}: {count} 个样本")
    unique_classes, counts = np.unique(y_val, return_counts=True)
    for class_label, count in zip(unique_classes, counts):
        print(f"y_val {class_label}: {count} 个样本")

    num_class = y_train.max() + 1
    noisy_y_train, _ = noisify(y_train, num_class, noise_type, noise_rate, seed)
    right_idx = np.where(y_train == noisy_y_train)[0]
    wrong_idx = np.where(y_train != noisy_y_train)[0]

    count = Counter(noisy_y_train[right_idx])
    print('clean_y_train: ', count)
    count = Counter(noisy_y_train[wrong_idx])
    print('noisy_y_trains: ', count)


    '''
    扩充维度,并转为dataloader
    '''
    x_train = np.expand_dims(x_train, 1)#扩充维度，（302400，1000）->302400, 1, 1000）
    x_val = np.expand_dims(x_val, 1)#扩充维度，（151200，1000）->(151200, 1, 1000)
    dataset_train = MyDataSet_train(x_train, y_train, noisy_y_train)#创建MyDataset实例
    dataset_test = MyDataSet(x_val, y_val)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)#数据分批次
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_test