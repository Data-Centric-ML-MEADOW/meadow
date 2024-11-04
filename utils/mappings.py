from typing import Any
from torchvision import transforms

import lightning as L
from models.base_pretrained import PreTrainedResNet

# a mapping from model identifier to the model class
MODEL_MAP: dict[str, Any] = {
    "resnet": PreTrainedResNet,
}

# a model may have a default set of transformations to apply to input data
TFMS_MAP: dict[str, transforms.Compose] = {
    "resnet": transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    "default": transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ]),
}
