import torch
from torchvision.transforms import functional as F

import numpy as np
from PIL import Image, ImageFilter
import torchvision
from torchvision import transforms
# import os


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor():
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize():
    def __init__(self, mean, std, to_bgr=True):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def __call__(self, image, target):
        if self.to_bgr:
            image = image[[2, 1, 0]]
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class GaussianBlur():
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma_min, sigma_max):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img, target):
        img = transforms.ToPILImage()(img)
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img, target


class ColorJitter():
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target

class Solarization():
    """Solarization augmentation in BYOL https://arxiv.org/abs/2006.07733."""

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, img, target):
        img = np.array(img)
        img = np.where(img < self.threshold, img, 255 -img)
        return Image.fromarray(img.astype(np.uint8)), target

'''
Inserted by Han 2021.8.5 11:07
'''
class Lighting():
    """Lighting noise(AlexNet - style PCA - based noise)."""

    _IMAGENET_PCA = {
        'eigval':
        torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec':
        torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }

    def __init__(self):
        self.alphastd = 0.1
        self.eigval = self._IMAGENET_PCA['eigval']
        self.eigvec = self._IMAGENET_PCA['eigvec']

    def __call__(self, img, target):
        assert isinstance(img, torch.Tensor), \
            "Expect torch.Tensor, got {}".format(type(img))
        if self.alphastd == 0:
            return img, target

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img)), target
