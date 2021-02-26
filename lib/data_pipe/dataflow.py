import os
import random
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.203, 0.1994, 0.2010]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

FACE_MEAN = [0.5, 0.5, 0.5]
FACE_STD = [0.5, 0.5, 0.5]


def get_test_loader(test_dataset_name, test_data_path):
    # Turn test_dataset str into list
    test_dataset_name = test_dataset_name.replace(" ", "")
    test_dataset_name = test_dataset_name.split(",")

    test_dataset = {}
    for name in test_dataset_name:
        path_to_test_data = os.path.join(test_data_path, name)

        data = np.load(path_to_test_data, "{}_data.npy".format(name))
        label = np.load(path_to_test_data, "{}_list.npy".format(name))

        test_dataset[dataset] = {
            "data": data,
            "label": label
        }

    return test_dataset


def get_train_loader(
        dataset,
        data_path,
        batch_size,
        num_workers,
        train_portion=1):
    train_transform = _get_train_transform(dataset)
    train_data = _get_dataset(dataset, data_path, train_transform)

    if train_portion != 1:
        train_len = len(train_data)
        indices = list(range(train_len))
        random.shuffle(indices)
        split = int(np.floor(train_portion * train_len))
        train_idx, val_idx = indices[:split], indices[split:]

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = _build_loader(
            train_data,
            True,
            batch_size,
            num_workers,
            sampler=train_sampler)
        val_loader = _build_loader(
            train_data,
            False,
            batch_size,
            num_workers,
            sampler=val_sampler)

        return train_loader, val_loader
    else:
        train_loader = _build_loader(
            train_data,
            True,
            batch_size,
            num_workers,
            sampler=None)

        return train_loader


def _build_loader(dataset, shuffle, batch_size, num_workers, sampler=None):
    if sampler is not None:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_workers,
            sampler=sampler
        )
    else:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_workers,
            shuffle=shuffle,
        )


def _get_dataset(dataset, data_path, transform, train=True):
    if dataset == "cifar10":
        return datasets.CIFAR10(root=data_path, train=train,
                                download=True, transform=transform)

    elif dataset == "cifar100":
        return datasets.CIFAR100(root=data_path, train=train,
                                 download=True, transform=transform)

    elif dataset == "imagenet":
        return datasets.ImageFolder(root=data_path, transform=transform)

    elif dataset == "casia":
        return datasets.ImageFolder(root=data_path, transform=transform)

    else:
        raise


def _get_train_transform(dataset):
    if dataset[:5] == "cifar":
        # CIFAR transforms
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])

    elif dataset == "imagenet":
        # Imagenet transforms
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

    elif dataset == "casia":
        # Face transforms
        train_transform = transforms.Compose([
            transforms.Resize(112),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(FACE_MEAN, FACE_STD)
        ])

    return train_transform


def _get_test_transform(dataset):
    if dataset[:5] == "cifar":
        # CIFAR transforms
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])

    elif dataset == "imagenet":
        # Imagenet transforms
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

    return test_transform
