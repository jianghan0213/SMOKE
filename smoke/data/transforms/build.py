from smoke.data.transforms import transforms as T
import os
from smoke.config import cfg
from PIL import Image
from torchvision import transforms

def build_transforms(cfg, is_train=True):
    to_bgr = cfg.INPUT.TO_BGR

    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr=to_bgr
    )

    gaussian_blur = T.GaussianBlur(sigma_min=cfg.INPUT.SIGMA_MIN, sigma_max=cfg.INPUT.SIGMA_MAX)

    color_jitter = T.ColorJitter(brightness=cfg.INPUT.BRIGHTNESS, contrast=cfg.INPUT.CONTRAST,
                                 hue=cfg.INPUT.HUE, saturation=cfg.INPUT.SATURATION)

    solarization = T.Solarization(threshold=cfg.INPUT.THRESHOLD)

    lighting = T.Lighting()

    transform = T.Compose(
        [
            T.ToTensor(),
            normalize_transform,
            gaussian_blur,
            T.ToTensor(),
            # color_jitter
            # solarization,
            # lighting,
        ]
    )
    return transform

# test transforms
# img = Image.open(r'C:\Users\92035\Music\实习\Mono3D\数据增强开发\000597.png')
# transform = build_transforms(cfg)
# img1, tgt1 = transform(img, img)
# img1 = transforms.ToPILImage()(img1)
# print(type(img1))
# img1.show()