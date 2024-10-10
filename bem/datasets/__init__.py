import os
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10, CelebA, MNIST, ImageNet, ImageFolder
from torch.utils.data import TensorDataset
from .lsun import LSUN
import torch.utils.data
from torch.utils.data import Subset, Dataset
import pickle
import numpy as np
from PIL import Image

from .Data import Generator

import argparse
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def inf_train_gen(img_name, data_size):
    def gen_data_from_img(image_mask, train_data_size):
        def sample_data(train_data_size):
            inds = np.random.choice(
                int(probs.shape[0]), int(train_data_size), p=probs)
            m = means[inds] 
            samples = np.random.randn(*m.shape) * std + m 
            return samples
        img = image_mask
        h, w = img.shape
        xx = np.linspace(-4, 4, w)
        yy = np.linspace(-4, 4, h)
        xx, yy = np.meshgrid(xx, yy)
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)
        means = np.concatenate([xx, yy], 1) # (h*w, 2)
        img = img.max() - img
        probs = img.reshape(-1) / img.sum() 
        std = np.array([8 / w / 2, 8 / h / 2])
        full_data = sample_data(train_data_size)
        return full_data
    image_mask = np.array(Image.open(f'{img_name}.png').rotate(
        180).transpose(0).convert('L'))
    dataset = gen_data_from_img(image_mask, data_size)
    return dataset / 4



class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


image_datasets = [
    "CIFAR10",
    "MINI_CIFAR10",
    "MNIST",
    "CELEBA",
    "CELEBA_HQ",
    "LSUN",
    "FFHQ",
]

toy_datasets = [
    "rose",
    "fractal_tree",
    'olympic_rings',
    'checkerboard'
]
def is_image_dataset(name):
    return name.upper() in [x.upper() for x in image_datasets]

def affine_transform(x):
    return 2*x -1

def inverse_affine_transform(x):
    return (x + 1) / 2

DATA_PATH = './data'

def get_dataset(p):

    # retrocompatibility with LIM original code
    config = dict2namespace(p)

    assert (config.data.dataset.upper() in [x.upper() for x in image_datasets]) \
        or (config.data.dataset.upper() in [x.upper() for x in Generator.available_distributions]) \
        or (config.data.dataset.lower() in [x.lower() for x in toy_datasets]), \
        "Dataset not available: {}.\nCan choose from:\n(Image)\t{}\n(Toy)\t{}\n(2d data)\t{}".format(config.data.dataset,
                                                                                          image_datasets,
                                                                                          toy_datasets,
                                                                                          Generator.available_distributions)
    
    if config.data.dataset.lower() in [x.lower() for x in Generator.available_distributions]:
        # for the moment, only supports dim <= 2
        assert config.data.dim <= 2, 'Only supports at most N-D data with N <= 2 for the moment. Please change Data.generator and Data.Distributions objects.'

        data_gen = Generator(config.data.dataset.lower(), 
                                    n = int(np.sqrt(config.data.n_mixture)) if config.data.dataset.split('_')[-1] == 'grid' else config.data.n_mixture, 
                                    std = config.data.std, 
                                    normalize = config.data.normalized,
                                    weights = config.data.weights,
                                    theta = config.data.theta,
                                    alpha = config.data.data_alpha,
                                    isotropic = config.data.isotropic,
                                    between_minus_1_1 = config.data.between_minus_1_1,
                                    quantile_cutoff = config.data.quantile_cutoff,
                                    )
        data_gen.generate(n_samples = config.data.nsamples)
        # possibly remove a dimension if nfeatures == 1
        if config.data.dim == 1:
            data_gen.samples = data_gen.samples[:, 0].unsqueeze(1)
        # add channel
        data_gen.samples = data_gen.samples.unsqueeze(1)
        import copy
        # train_data
        dataset = TensorDataset(copy.deepcopy(data_gen.samples), torch.tensor([0.]).repeat(data_gen.samples.shape[0]))
        #dataset = TensorDataset(data_gen.samples)
        data_gen.generate(n_samples = config.data.nsamples)
        # possibly remove a dimension if nfeatures == 1
        if config.data.dim == 1:
            data_gen.samples = data_gen.samples[:, 0].unsqueeze(1)
        # add channel
        data_gen.samples = data_gen.samples.unsqueeze(1)
        #test_data
        test_dataset = TensorDataset(copy.deepcopy(data_gen.samples), torch.tensor([0.]).repeat(data_gen.samples.shape[0]))
        return dataset, test_dataset


    if config.data.dataset.lower() in [x.lower() for x in toy_datasets]:
        xraw = inf_train_gen(os.path.join(DATA_PATH, 
                                          'toy', 
                                          "img_{}".format(config.data.dataset.lower())),
                              config.data.nsamples)
        xte = inf_train_gen(os.path.join(DATA_PATH, 
                                          'toy', 
                                          "img_{}".format(config.data.dataset.lower())),
                              config.data.nsamples)
        xraw = torch.from_numpy(xraw).float().unsqueeze(1)
        xte = torch.from_numpy(xte).float().unsqueeze(1)
        dataset = torch.utils.data.TensorDataset(xraw, torch.tensor([0.]).repeat(xraw.shape[0]))
        test_dataset = torch.utils.data.TensorDataset(xte, torch.tensor([0.]).repeat(xte.shape[0]))
        return dataset, test_dataset
    
    config.data.dataset = config.data.dataset.upper()
    
    if config.data.dataset == "CIFAR10":
        if config.data.random_flip is False:
            tran_transform = test_transform = transforms.Compose(
                [transforms.Resize(config.data.image_size), transforms.ToTensor(), transforms.Lambda(affine_transform)]
            )
        else:
            tran_transform = transforms.Compose(
                [
                    transforms.Resize(config.data.image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Lambda(affine_transform),
                ]
            )
        dataset = CIFAR10(
            os.path.join(DATA_PATH, "cifar10"),
            train=True,
            download=True,
            transform=tran_transform,
        )
        test_dataset = CIFAR10(
            os.path.join(DATA_PATH, "cifar10_test"),
            train=False,
            download=True,
            transform=tran_transform,#test_transform,
        )
    elif config.data.dataset == "MINI_CIFAR10":
        dataset = ImageFolder(
                root=os.path.join(DATA_PATH, "mini_cifar10"),
                transform = transforms.Compose(
                    [transforms.RandomHorizontalFlip(p=0.5),
                     transforms.ToTensor(),
                     transforms.Resize(config.data.image_size),
                     transforms.Lambda(affine_transform)]
                ),
            )
        test_dataset = ImageFolder(
                root=os.path.join(DATA_PATH, "mini_cifar10"),
                transform = transforms.Compose(
                    [transforms.RandomHorizontalFlip(p=0.5),
                     transforms.ToTensor(),
                     transforms.Resize(config.data.image_size),
                     transforms.Lambda(affine_transform)]
                ),
            )
    
    elif config.data.dataset == "MNIST":
        mnist_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.CenterCrop(config.data.image_size),
                transforms.ToTensor(),
                transforms.Lambda(affine_transform),
            ]
        )
        dataset = MNIST(
            os.path.join(DATA_PATH, "mnist"),
            train=True,
            download=True,
            transform=mnist_transform,
        )
        test_dataset = MNIST(
            os.path.join(DATA_PATH, "mnist_test"),
            train=False,
            download=True,
            transform=mnist_transform, #test_transform,
        )
        
    elif config.data.dataset == "CELEBA":
        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64
        if config.data.random_flip:
            dataset = CelebA(
                root=os.path.join(DATA_PATH),#, "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(config.data.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Lambda(affine_transform)
                    ]
                ),
                download=True,
            )
        else:
            dataset = CelebA(
                root=os.path.join(DATA_PATH),#, "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(config.data.image_size),
                        transforms.ToTensor(),
                        transforms.Lambda(affine_transform)
                    ]
                ),
                download=True,
            )

        test_dataset = CelebA(
            root=os.path.join(DATA_PATH),#, "celeba"),
            split="test",
            transform=transforms.Compose(
                [
                    Crop(x1, x2, y1, y2),
                    transforms.Resize(config.data.image_size),
                    transforms.ToTensor(),
                    transforms.Lambda(affine_transform)
                ]
            ),
            download=True,
        )
    
    elif config.data.dataset == "CELEBA_HQ":
        if config.data.random_flip:
            dataset = ImageFolder(
                root=os.path.join(DATA_PATH, "celebahq"),
                transform = transforms.Compose(
                    [transforms.RandomHorizontalFlip(p=0.5),
                     transforms.ToTensor(),
                     transforms.Resize(config.data.image_size),
                     transforms.Lambda(affine_transform)]
                ),
            )
        else:
            dataset = ImageFolder(
                root=os.path.join(DATA_PATH, "celebahq"),
                transform = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Resize(config.data.image_size),
                     transforms.Lambda(affine_transform)]
                ),
            )

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = (
            indices[: int(num_items * 0.9)],
            indices[int(num_items * 0.9):],
        )
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)        

    elif config.data.dataset == "LSUN":
        train_folder = "{}_train".format(config.data.category)
        val_folder = "{}_val".format(config.data.category)
        if config.data.random_flip:
            dataset = LSUN(
                root=os.path.join(DATA_PATH, "lsun"),
                classes=[train_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize(config.data.image_size),
                        transforms.CenterCrop(config.data.image_size),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor(),
                        transforms.Lambda(affine_transform)
                    ]
                ),
            )
        else:
            dataset = LSUN(
                root=os.path.join(DATA_PATH, "lsun"),
                classes=[train_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize(config.data.image_size),
                        transforms.CenterCrop(config.data.image_size),
                        transforms.ToTensor(),
                        transforms.Lambda(affine_transform)
                    ]
                ),
            )

        test_dataset = LSUN(
            root=os.path.join(DATA_PATH, "lsun"),
            classes=[val_folder],
            transform=transforms.Compose(
                [
                    transforms.Resize(config.data.image_size),
                    transforms.CenterCrop(config.data.image_size),
                    transforms.ToTensor(),
                    transforms.Lambda(affine_transform)
                ]
            ),
        )

    elif config.data.dataset == "FFHQ":
        if config.data.random_flip:
            dataset = FFHQ(
                path=os.path.join(DATA_PATH, "FFHQ"),
                transform=transforms.Compose(
                    [transforms.RandomHorizontalFlip(p=0.5), 
                     transforms.ToTensor(), 
                     transforms.Lambda(affine_transform)]
                ),
                resolution=config.data.image_size,
            )
        else:
            dataset = FFHQ(
                path=os.path.join(DATA_PATH, "FFHQ"),
                transform=transforms.ToTensor(),
                resolution=config.data.image_size,
            )

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = (
            indices[: int(num_items * 0.9)],
            indices[int(num_items * 0.9) :],
        )
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)
    else:
        dataset, test_dataset = None, None

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = (2. * X - 1.0)*5.
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 5.0) / 10.0

    return torch.clamp(X, 0.0, 1.0)

class imagenet64_dataset(Dataset):
    """`DownsampleImageNet`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    train_list = [
        ['train_data_batch_1'],
        ['train_data_batch_2'],
        ['train_data_batch_3'],
        ['train_data_batch_4'],
        ['train_data_batch_5'],
        ['train_data_batch_6'],
        ['train_data_batch_7'],
        ['train_data_batch_8'],
        ['train_data_batch_9'],
        ['train_data_batch_10']
    ]
    test_list = [
        ['val_data'],
    ]

    def __init__(self, root, train=True):
        self.root = os.path.expanduser(root)
        self.transform = transforms.ToTensor()
        self.train = train  # training set or test set

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()
            # resize label range from [1,1000] to [0,1000),
            # This is required by CrossEntropyLoss
            self.train_labels[:] = [x - 1 for x in self.train_labels]

            self.train_data = np.concatenate(self.train_data)
            [picnum, pixel] = self.train_data.shape
            pixel = int(np.sqrt(pixel / 3))
            self.train_data = self.train_data.reshape((picnum, 3, pixel, pixel))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            [picnum,pixel]= self.test_data.shape
            pixel = int(np.sqrt(pixel/3))

            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()

            # resize label range from [1,1000] to [0,1000),
            # This is required by CrossEntropyLoss
            self.test_labels[:] = [x - 1 for x in self.test_labels]
            self.test_data = self.test_data.reshape((picnum, 3, pixel, pixel))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, y_data = self.train_data[index], self.train_labels[index]
        else:
            img, y_data = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        x_data = self.transform(img)
        y_data = torch.tensor(y_data, dtype=torch.int64)

        return x_data, y_data

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)