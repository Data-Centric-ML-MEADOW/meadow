import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy
from torchvision import models


class PreTrainedResNet(L.LightningModule):
    resnet_variant_map = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
    }

    def __init__(
        self, out_classes, resnet_variant=18, optimizer=torch.optim.Adam, lr=1e-2
    ):
        if resnet_variant not in self.resnet_variant_map:
            raise ValueError("Invalid ResNet variant argument!")
        super().__init__()
        self.save_hyperparameters()

        self.out_classes = out_classes
        self.resnet_variant = resnet_variant

        self.optimizer = optimizer
        self.lr = lr

        # accuracy metric for train/val loop
        self.accuracy = Accuracy(task="multiclass", num_classes=self.out_classes)

        # download pretrained resnet
        backbone = self.resnet_variant_map[self.resnet_variant](weights="DEFAULT")
        resnet_num_ftrs = backbone.fc.in_features

        # extract resnet CNN/feat extraction layers
        layers = list(backbone.children())[:-1]
        self.resnet_feat_extractor = torch.nn.Sequential(*layers)
        self.resnet_feat_extractor.eval()  # freeze resnet backbone
        self.resnet_feat_extractor.requires_grad_(False)

        # final layer to perform classification
        self.fc = torch.nn.Linear(resnet_num_ftrs, self.out_classes)

    def forward(self, x):
        with torch.no_grad():
            repr = self.resnet_feat_extractor(x).flatten(1)
        x = self.fc(repr)
        return x

    def _batch_step(self, batch, batch_kind):
        if batch_kind == "train":
            self.fc.train()
        else:
            self.fc.eval()
        x, y, metadata = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        # logging onto tensorboard
        self.log(f"{batch_kind}_loss", loss, prog_bar=True)
        self.log(f"{batch_kind}_acc_f1", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._batch_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._batch_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._batch_step(batch, "test")

    def predict_step(self, batch, batch_idx):
        self.eval()
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer
