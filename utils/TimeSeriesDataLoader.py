from torch.utils.data.dataset import Dataset
from torchvision import transforms
import csv
import torch
class TimeSeriesDataset(Dataset):
    def __init__(self, file_path, transform = None):
        self.labels = []
        self.features = []
        with open(file_path) as input_file:
            reader = csv.reader(input_file, quoting=csv.QUOTE_NONNUMERIC)
            [self.labels.append(int(row[0] -1)) for row_number, row in enumerate(reader)]

        with open(file_path) as input_file2:
            reader2 = csv.reader(input_file2, quoting=csv.QUOTE_NONNUMERIC)
            for row_number, row in enumerate(reader2):
                row.pop(0)
                self.features.append(row)

        self.data_len = len(self.features)
        self.to_tensor = transforms.ToTensor()
        self.transform = transform

    def __getitem__(self, index):

        data = self.features[index]
        if self.transform is not None:
            data = torch.FloatTensor(data)
            #data = self.transform(data)

        label = self.labels[index]
        return data, label

    def __len__(self):
        return self.data_len  # of how many data(images?) you have

