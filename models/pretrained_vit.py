import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, MulticlassF1Score
from torchvision import models


class PreTrainedViT(L.LightningModule):
    vit_variant_map = {
        "B_16": "vit_base_patch16_224",
    }

    def __init__(
        self,
        out_classes,
        variant="B_16",
        optimizer=torch.optim.AdamW,  # type: ignore
        lr=1e-4,
        freeze_backbone=True,
    ):
        if variant not in self.vit_variant_map:
            raise ValueError("Invalid ViT variant argument!")
        super().__init__()
        self.save_hyperparameters()

        self.out_classes = out_classes
        self.vit_variant = variant

        self.optimizer = optimizer
        self.lr = lr
        self.freeze_backbone = freeze_backbone

        # accuracy metric for train/val loop
        self.accuracy = Accuracy(task="multiclass", num_classes=self.out_classes)
        self.f1_score = MulticlassF1Score(num_classes=self.out_classes, average="macro")

        # download pretrained ViT
        self.vit = torch.hub.load("facebookresearch/dino:main", self.vit_variant, pretrained=True)
        vit_num_ftrs = self.vit.head.in_features

        # extract ViT feat extraction layers
        self.vit_feat_extractor = torch.nn.Sequential(self.vit.head)

        # final layer to perform classification
        self.fc = torch.nn.Linear(vit_num_ftrs, self.out_classes)

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                repr = self.vit_feat_extractor(x).flatten(1)
        else:
            repr = self.vit_feat_extractor(x).flatten(1)
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
        f1 = self.f1_score(y_hat, y)
        # logging onto tensorboard
        self.log(f"{batch_kind}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{batch_kind}_acc", acc, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log(f"{batch_kind}_f1", f1, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._batch_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._batch_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._batch_step(batch, "test")

    def predict_step(self, batch, batch_idx):
        self.eval()
        x, _, metadata = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer
