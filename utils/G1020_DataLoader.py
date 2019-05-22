from torch.utils.data.dataset import Dataset
from torchvision import transforms
from scipy.misc import imread, imresize
import numpy as np
from PIL import Image
import cv2 as cv

class G1020_Dataset(Dataset):
    def __init__(self, file_path,imageSize, transform = None):

        self.height = imageSize
        self.width = imageSize
        self.transform = transform
        self.lines = [line.rstrip('\n') for line in open(file_path)]
        self.paths = [line.split()[0] for line in self.lines]
        self.labels = []
        [self.labels.append(int(line.split()[1])) for line in self.lines]
        self.to_tensor = transforms.ToTensor()
        self.data_len = len(self.paths)

    def __getitem__(self, index):

        img = imread(self.paths[index])
        img = imresize(img, (self.height, self.width , 3))

        img[0] = img[0] - np.mean(img[0])
        img[1] = img[1] - np.mean(img[1])
        img[2] = img[2] - np.mean(img[2])

        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[index]
        return img, label

    def __len__(self):
        return self.data_len  # of how many data(images?) you have

