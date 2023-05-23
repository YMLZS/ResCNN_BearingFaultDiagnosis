import torch
import pandas as pd
from torch.utils.data import Dataset
from einops import rearrange

class MyDataset(Dataset):
    def __init__(self, path, task, transform=None, target_transform=None, loader=None):
        # 读取csv文件
        dataFrame = pd.read_csv(path, encoding='utf-8', header=None)
        martrix = dataFrame.values
        datas = martrix[:, :-3]
        if task == 'fd':
            labels = martrix[:, -3]
        elif task == 'loc':
            labels = martrix[:, -2]
        elif task == 'dia':
            labels = martrix[:, -1]
        elif task == 'multi':
            labels = martrix[:, -3:]
        self.datas = torch.Tensor(datas)
        self.labels = torch.Tensor(labels)
        self.loader = loader

    def __getitem__(self, index):
        # 获取数据和标签
        data = self.datas[index]
        label = self.labels[index]
        # 处理数据
        data = data.unsqueeze(0)
        return data, label

    def __len__(self):
        return len(self.datas)
