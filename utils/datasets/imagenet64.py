import torch
from torchvision import transforms as tf
import pickle
import tqdm
import itertools
import numpy as np

tf_imagenet64_train = tf.Compose([
    tf.RandomHorizontalFlip(),
    tf.RandomVerticalFlip(),
    tf.RandomRotation(degrees=10),
    tf.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

tf_imagenet64_test = tf.Compose([
    tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class ImageNet64():
    training_files = [f'train_data_batch_{i}' for i in range(1, 11)]
    test_files = ['val_data']
    classes = {f'{i}': f'{i}' for i in range(1000)}

    def __init__(self, root: str, train: bool = True, transform=None):

        self.transform = transform
        self.train = train

        self.data = []

        if train:
            for file in tqdm.tqdm(self.training_files, total=len(self.training_files)):
                with open(root + file, 'rb') as fd:
                    self.data.append(pickle.load(fd))
        else:
            for file in self.test_files:
                with open(root + file, 'rb') as fd:
                    self.data.append(pickle.load(fd))

        self.targets = list(itertools.chain.from_iterable([item['labels'] for item in self.data]))
        self.targets = [item - 1 for item in self.targets]
        self.idxs = []

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int):
        target = self.targets[index]
        if index == 0:
            index1 = index2 = 0
        else:
            if index >= 10*128116:
                index1, index2 = 9, index - 10*128116
            else:
                index1, index2 = np.divmod(index, 128116)
        img = self.data[index1]['data'][index2]
        img = torch.tensor(img, dtype=torch.float32)

        img = img/255.0
        img = img.reshape((3, 64, 64))
        if self.transform is not None:
            img = self.transform(img)
        return img, target
