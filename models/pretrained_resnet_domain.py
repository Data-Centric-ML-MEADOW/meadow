import torch.nn.functional as F

from models.base_pretrained import PreTrainedResNet


class PreTrainedResNetDomain(PreTrainedResNet):
    """Pretrained ResNet for domain mapping."""

    # swap the batch step helper function to use domain label instead of class label
    def _batch_step(self, batch, batch_kind):
        if batch_kind == "train":
            self.train()
        else:
            self.eval()
        x, _, metadata = batch
        # evaluate against domain label instaed of class label
        d = metadata[:, 0]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, d)
        acc = self.accuracy(y_hat, d)
        f1 = self.f1_score(y_hat, d)
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

