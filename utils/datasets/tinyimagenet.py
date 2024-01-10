# from https://gist.github.com/z-a-f/b862013c0dc2b540cf96a123a6766e54

import imageio
import numpy as np
import os

from collections import defaultdict
from torch.utils.data import Dataset

from tqdm.autonotebook import tqdm
from torchvision import transforms as tf

tf_tinyimagenet_train = tf.Compose([
    tf.ToTensor(),
    tf.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
    tf.RandomHorizontalFlip(),
    tf.RandomVerticalFlip()])

tf_tinyimagenet_test = tf.Compose([
    tf.ToTensor()])


def download_and_unzip(URL, root_dir):
    error_message = 'Download is not yet implemented. Please, go to {URL} yourself.'
    raise NotImplementedError(error_message.format(URL))


def _add_channels(img, total_channels=3):
    while len(img.shape) < 3:  # third axis is the channels
        img = np.expand_dims(img, axis=-1)
    while (img.shape[-1]) < 3:
        img = np.concatenate([img, img[:, :, -1:]], axis=-1)
    return img


class TinyImageNetPaths:
    def __init__(self, root_dir, download=False):
        if download:
            download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip', root_dir)

        train_path = os.path.join(root_dir, 'train')
        val_path = os.path.join(root_dir, 'val')

        wnids_path = os.path.join(root_dir, 'wnids.txt')
        words_path = os.path.join(root_dir, 'words.txt')

        self._make_paths(train_path, val_path, wnids_path, words_path)

    def _make_paths(self, train_path, val_path,
                    wnids_path, words_path):
        self.ids = []
        with open(wnids_path, 'r') as idf:
            for nid in idf:
                nid = nid.strip()
                self.ids.append(nid)
        self.nid_to_words = defaultdict(list)
        with open(words_path, 'r') as wf:
            for line in wf:
                nid, labels = line.split('\t')
                labels = list(map(lambda x: x.strip(), labels.split(',')))
                self.nid_to_words[nid].extend(labels)

        self.paths = {
            'train': [],  # [img_path, id, nid, box]
            'val': [],  # [img_path, id, nid, box]
            'test': []  # img_path
            }

        # Get the validation paths and labels
        with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
            for line in valf:
                fname, nid, x0, y0, x1, y1 = line.split()
                fname = os.path.join(val_path, 'images', fname)
                bbox = int(x0), int(y0), int(x1), int(y1)
                label_id = self.ids.index(nid)
                self.paths['val'].append((fname, label_id, nid, bbox))

        # Get the training paths
        train_nids = os.listdir(train_path)
        for nid in train_nids:
            anno_path = os.path.join(train_path, nid, nid+'_boxes.txt')
            imgs_path = os.path.join(train_path, nid, 'images')
            label_id = self.ids.index(nid)
            with open(anno_path, 'r') as annof:
                for line in annof:
                    fname, x0, y0, x1, y1 = line.split()
                    fname = os.path.join(imgs_path, fname)
                    bbox = int(x0), int(y0), int(x1), int(y1)
                    self.paths['train'].append((fname, label_id, nid, bbox))


'''Datastructure for the tiny image dataset.
Args:
  root_dir: Root directory for the data
  mode: One of 'train', 'test', or 'val'
  preload: Preload into memory
  load_transform: Transformation to use at the preload time
  transform: Transformation to use at the retrieval time
  download: Download the dataset
Members:
  tinp: Instance of the TinyImageNetPaths
  img_data: Image data
  targets: Label data
'''


class TinyImageNetDataset(Dataset):

    def __init__(self, root, train=True, transform=None):
        tinp = TinyImageNetPaths(root + '/tiny-imagenet-200', download=False)
        self.mode = 'train' if train else 'val'
        self.label_idx = 1  # from [image, id, nid, box]
        self.preload = True
        self.transform = transform
        self.transform_results = dict()
        load_transform = None
        self.train = train

        self.classes = {}
        for i in range(200):
            self.classes.update({f'{i}' : f'{i}'})

        self.IMAGE_SHAPE = (64, 64, 3)

        self.img_data = []
        self.targets = []

        self.max_samples = None
        self.samples = tinp.paths[self.mode]
        self.samples_num = len(self.samples)

        if self.max_samples is not None:
            self.samples_num = min(self.max_samples, self.samples_num)
            self.samples = np.random.permutation(self.samples)[:self.samples_num]

        if self.preload:
            load_desc = 'Preloading {} data...'.format(self.mode)
            self.img_data = np.zeros((self.samples_num,) + self.IMAGE_SHAPE,
                                dtype=np.float32)
            self.targets = np.zeros((self.samples_num,), dtype=np.int64)
            for idx in tqdm(range(self.samples_num), desc=load_desc):
                s = self.samples[idx]
                img = imageio.imread(s[0])
                img = _add_channels(img)
                self.img_data[idx] = img
                self.targets[idx] = s[self.label_idx]

        if load_transform:
            for lt in load_transform:
                result = lt(self.img_data, self.targets)
                self.img_data, self.targets = result[:2]
                if len(result) > 2:
                    self.transform_results.update(result[2])

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        if self.preload:
            img = self.img_data[idx]
            lbl = self.targets[idx]
        else:
            s = self.samples[idx]
            img = imageio.imread(s[0])
            img = _add_channels(img)
            lbl = s[self.label_idx]

        if self.transform is not None:
            img = self.transform(img)
        return img, lbl
