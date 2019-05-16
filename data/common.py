from torch.utils.data import Dataset
import torch
import numpy as np


def get_index_of_classes(target, classes):
    l = []

    if isinstance(classes, int):  # if only one class is given, make it a list
        classes = [classes]

    for cl in classes:
        l.append(torch.nonzero(target == cl).squeeze())
    return torch.cat(l)


def split_dataset(dataset_size, shuffle, validation_split=0.2, test_split=None, batch_size=128):

    # Creating data indices for training and validation splits:
    indices = list(range(dataset_size))
    split1 = int(np.floor(validation_split * dataset_size))

    if shuffle:
        np.random.shuffle(indices)

    if test_split is not None:
        split2 = int(np.floor(test_split * dataset_size))
        split1 += split2

        return indices[split1:], indices[split2:split1], indices[: split2]
    elif validation_split == 0:
        return indices, indices[:batch_size]  # just for not being empty and avoid a complex code
    else:
        return indices[split1:], indices[:split1]


class DoubleDataset(torch.utils.data.Dataset):

    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, index):
        if index >= len(self.dataset1):
            data_1 = self.dataset1[index % len(self.dataset1)]
        else:
            data_1 = self.dataset1[index]
        if index >= len(self.dataset2):
            data_2 = self.dataset2[index % len(self.dataset2)]
        else:
            data_2 = self.dataset2[index]

        return data_1, data_2

    def __len__(self):
        return max(len(self.dataset1), len(self.dataset2))
