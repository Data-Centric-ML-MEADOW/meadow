from typing import Any

from torchvision import transforms

from models.base_pretrained import PreTrainedResNet

# a mapping from model identifier to the model class
MODEL_MAP: dict[str, Any] = {
    "resnet": PreTrainedResNet,
}

# a model may have a default set of transformations to apply to input data
TFMS_MAP: dict[str, transforms.Compose] = {
    "resnet": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "resnet-tfms": transforms.Compose(
        [
            transforms.ToTensor(),
            # random augmentations for training more robustly
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30, 30)),
            transforms.RandomErasing(p=0.1),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "default": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((448, 448)),
        ]
    ),
}
