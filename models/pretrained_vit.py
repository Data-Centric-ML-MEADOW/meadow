import lightning as L
from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_b_32, ViT_B_32_Weights
import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, MulticlassF1Score

class PreTrainedViT(L.LightningModule):
    vit_variant_map = {
        16: (vit_b_16, ViT_B_16_Weights.DEFAULT),
        32: (vit_b_32, ViT_B_32_Weights.DEFAULT),
    }

    def __init__(
        self,
        out_classes,
        variant=16,
        optimizer=torch.optim.AdamW,  # type: ignore
        lr=1e-4,
        freeze_backbone=True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Validate variant
        try:
            variant = int(variant)
        except ValueError:
            raise ValueError(f"Invalid ViT variant: {variant}. It must be one of {list(self.vit_variant_map.keys())}")

        if variant not in self.vit_variant_map:
            raise ValueError(f"Invalid ViT variant! Choose from {list(self.vit_variant_map.keys())}")

        self.vit_variant = variant
        self.out_classes = out_classes
        self.optimizer = optimizer
        self.lr = lr
        self.freeze_backbone = freeze_backbone

        # Initialize ViT with pretrained weights
        vit_model, vit_weights = self.vit_variant_map[self.vit_variant]
        self.vit = vit_model(weights=vit_weights)

        # Modify the classification head to match the number of output classes
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = torch.nn.Linear(in_features, out_classes)

        # Metrics for evaluation
        self.accuracy = Accuracy(task="multiclass", num_classes=out_classes)
        self.f1_score = MulticlassF1Score(num_classes=out_classes, average="macro")

        # Optionally freeze the ViT encoder
        if self.freeze_backbone:
            self.vit.encoder.requires_grad_(False)
            self.vit.encoder.eval()

    def forward(self, x):
        """Forward pass through the Vision Transformer."""
        return self.vit(x)

    def _batch_step(self, batch, batch_kind):
        """Process a single batch."""
        x, y, metadata = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        f1 = self.f1_score(y_hat, y)

        # Log metrics
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
        """Make predictions for a batch."""
        x, _, metadata = batch
        return self(x)

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer
