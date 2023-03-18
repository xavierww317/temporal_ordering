import os
import numpy as np
from PIL import Image
import torch
import torchvision.datasets
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.datasets import ImageFolder,DatasetFolder
import torchvision.transforms as transforms
from torch.utils.data.sampler import SequentialSampler

transform = transforms.Compose(
    [
        # transforms.Resize((129,3300))
        # transforms.RandomCrop(224)
        # transforms.ToTensor()
    ]
)

root_folder = './picture'

class Movie_DataSet(Dataset):
    def __init__(self, root_path, transform):
        data_picture = []
        for root, dirs, files in os.walk(root_path):
            for i in range(len(files)):
                file_str = './picture/Interstellar-{:010d}.jpeg'.format(i+1)
                data_picture.append(file_str)

        self.data_picture = data_picture
        self.transform = transform

    def __getitem__(self, item):
        data_path = self.data_picture[item]
        print(data_path)
        data = torch.from_numpy(np.transpose(np.array(Image.open(data_path)), (2, 0, 1))).type(torch.FloatTensor)

        return data, 0

    def __len__(self):
        return len(self.data_picture)

def loaddata(root, batchsize=10, num_workers=0):
    data = Movie_DataSet(root, transform=transform)
    data_sample = SequentialSampler(data)
    data_iter = DataLoader(data, batch_size=batchsize, shuffle=False, num_workers=num_workers, sampler=data_sample)
    return data_iter

if __name__ == '__main__':
    data_loader = loaddata(root_folder, 1)
    for img, _ in iter(data_loader):
        print(img.shape, type(img))
        print(_)
        break