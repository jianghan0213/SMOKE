from . import transforms as T


def build_transforms(cfg, is_train=True):
    to_bgr = cfg.INPUT.TO_BGR

    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr=to_bgr
    )

    gaussian_blur = T.GaussianBlur(sigma_min=cfg.INPUT.SIGMA_MIN, sigma_max=cfg.SIGMA_MAX)

    color_jitter = T.ColorJitter(brightness=cfg.INPUT.BRIGHTNESS, contrast=cfg.INPUT.CONTRAST,
                                 hue=cfg.INPUT.HUE, saturation=cfg.INPUT.SATURATION)

    solarization = T.Solarization(threshold=cfg.THRESHOLD)

    lighting = T.Lighting()

    transform = T.Compose(
        [
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform
