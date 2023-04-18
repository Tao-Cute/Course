# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform




def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    nb_classes = args.num_classes
    return dataset, nb_classes


def build_transform(is_train, args):
    if is_train:
        t = []
        t.append(transforms.Resize(256))
        t.append(transforms.CenterCrop(224))
        if args.flip:
            t.append(transforms.RandomVerticalFlip(p = args.flip))
            t.append(transforms.RandomHorizontalFlip(p = args.flip))
        if args.rotation:
            t.append(transforms.RandomRotation(args.rotation))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        return transforms.Compose(t)

    t = []
    t.append(transforms.Resize(256))
    t.append(transforms.CenterCrop(224))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
