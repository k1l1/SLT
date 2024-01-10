import torch
from torchvision import transforms as tf


class femnist_to_cifar_format_transform:
    def __init__(self):
        pass

    def __call__(self, image):
        image = tf.functional.resize(image, [32, 32])
        image = torch.stack(3*[image])[:, 0, :, :]
        return image


tf_femnist_train = tf.Compose([
    femnist_to_cifar_format_transform(),
    tf.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.9, 1.1))])

tf_femnist_test = tf.Compose([
    femnist_to_cifar_format_transform()])


class FEMNIST():
    training_file = 'data'
    targets_file = 'targets'
    classes = {
        '0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7',
        '8': '8', '9': '9',

        '10': 'A', '11': 'B', '12': 'C', '13': 'D', '14': 'E',
        '15': 'F', '16': 'G', '17': 'H', '18': 'I', '19': 'J', '20': 'K', '21': 'L',
        '22': 'M', '23': 'N', '24': 'O', '25': 'P', '26': 'Q', '27': 'R', '28': 'S', '29': 'T',
        '30': 'U', '31': 'V', '32': 'W', '33': 'X', '34': 'Y', '35': 'Z',

        '36': 'a', '37': 'b', '38': 'c', '39': 'd', '40': 'e',
        '41': 'f', '42': 'g', '43': 'h', '44': 'i', '45': 'j', '46': 'k', '47': 'l',
        '48': 'm', '49': 'n', '50': 'o', '51': 'p', '52': 'q', '53': 'r', '54': 's', '55': 't',
        '56': 'u', '57': 'v', '58': 'w', '59': 'x', '60': 'y', '61': 'z'}

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform=None):

        self.transform = transform
        self.train = train

        if train:
            self.data = torch.load(
                root + "femnist/" + self.training_file + "_train.pt")[:, :, :640500]
            self.targets = torch.load(
                root + "femnist/" + self.targets_file + "_train.pt")[:640500]
        else:
            self.data = torch.load(root + "femnist/" +
                                   self.training_file + "_test.pt")
            self.targets = torch.load(
                root + "femnist/" + self.targets_file + "_test.pt")

    def __len__(self):
        return self.data.shape[2]

    def __getitem__(self, index: int):
        img, target = self.data[:, :, index].unsqueeze(
            0), self.targets[index].type(torch.LongTensor)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
