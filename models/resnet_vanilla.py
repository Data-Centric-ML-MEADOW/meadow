import torch
from torch import nn
from torchvision import models


class PreTrainedResNetVanilla(nn.Module):
    resnet_variant_map = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }

    def __init__(
        self,
        out_classes,
        variant=18,
        freeze_backbone=True,
        **kwargs,  # ignore all other arguments
    ):
        if isinstance(variant, str):
            if variant.isnumeric():
                variant = int(variant)
            else:
                raise ValueError("Invalid ResNet variant argument!")
        if variant not in self.resnet_variant_map:
            raise ValueError("Invalid ResNet variant argument!")
        super().__init__()

        self.out_classes = out_classes
        self.resnet_variant = variant

        self.freeze_backbone = freeze_backbone

        # download pretrained resnet
        backbone = self.resnet_variant_map[self.resnet_variant](weights="DEFAULT")
        resnet_num_ftrs = backbone.fc.in_features

        # extract resnet CNN/feat extraction layers
        layers = list(backbone.children())[:-1]
        self.resnet_feat_extractor = torch.nn.Sequential(*layers)
        if self.freeze_backbone:
            self.resnet_feat_extractor.eval()  # freeze resnet backbone
            self.resnet_feat_extractor.requires_grad_(False)

        # final layer to perform classification
        self.fc = torch.nn.Linear(resnet_num_ftrs, self.out_classes)

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                repr = self.resnet_feat_extractor(x).flatten(1)
        else:
            repr = self.resnet_feat_extractor(x).flatten(1)
        x = self.fc(repr)
        return x
