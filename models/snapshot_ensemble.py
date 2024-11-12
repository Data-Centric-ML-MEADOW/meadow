import math

import lightning as L
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import Accuracy, MulticlassF1Score


class SnapshotEnsemble(L.LightningModule):
    def __init__(
        self,
        out_classes: int,
        train_loader_len: int,
        base_model,
        base_model_args=None,
        num_estimators=5,
        optimizer=torch.optim.AdamW,  # type: ignore
        lr=0.2,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.optimizer = optimizer
        self.lr = lr
        self.out_classes = out_classes
        self.base_model = base_model
        self.base_model_args = base_model_args or {}
        self.num_estimators = num_estimators
        # must pass train_loader_len since self.trainer.num_training_batches is unreliable
        self.train_loader_len = train_loader_len

        # accuracy metric for train/val loop
        self.accuracy = Accuracy(task="multiclass", num_classes=self.out_classes)
        self.f1_score = MulticlassF1Score(num_classes=self.out_classes, average="macro")

        # create empty ensemble module list, will be populated in training
        self.ensemble = torch.nn.ModuleList()
        for _ in range(self.num_estimators):
            self.ensemble.append(self.base_model(**self.base_model_args))
        self.curr_estimator = 0

        # disable automatic optimization to allow for lr scheduling
        self.automatic_optimization = False

    def train(self, mode=True):
        super().train(mode)
        for estimator in self.ensemble:
            estimator.train(mode)

    def eval(self):
        super().eval()
        for estimator in self.ensemble:
            estimator.eval()

    def forward(self, x):
        # forward passes input through each model in ensemble, w/ soft (avg) voting strategy
        return torch.stack(
            [F.softmax(m(x), dim=-1) for m in self.ensemble], dim=0
        ).mean(dim=0)

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.curr_estimator = 0
        assert self.trainer.max_epochs is not None
        if self.trainer.max_epochs % self.num_estimators != 0:
            raise ValueError(
                f"Trainer.max_epochs ({self.trainer.max_epochs}) must be divisible by"
                f" SnapshotEnsemble.num_estimators ({self.num_estimators})."
            )

    def training_step(self, batch, batch_idx):
        x, y, metadata = batch
        self.train()
        assert self.trainer.max_epochs is not None
        # determine which estimator to train
        estimator_idx = self.current_epoch // (
            self.trainer.max_epochs // self.num_estimators
        )
        if estimator_idx != self.curr_estimator:
            # set next estimator as a snapshot of the last estimator
            self.ensemble[estimator_idx].load_state_dict(
                self.ensemble[self.curr_estimator].state_dict()
            )
            self.curr_estimator = estimator_idx
        estimator = self.ensemble[estimator_idx]
        y_hat = estimator(x)

        opt = self.optimizers()
        opt.zero_grad()  # type: ignore
        loss = F.cross_entropy(y_hat, y)
        self.manual_backward(loss)
        opt.step()  # type: ignore
        sch = self.lr_schedulers()
        sch.step()  # type: ignore

        acc = self.accuracy(y_hat, y)
        f1 = self.f1_score(y_hat, y)
        # logging onto tensorboard
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("train_f1", f1, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def _eval_step(self, batch, batch_kind):
        x, y, metadata = batch
        self.eval()
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        f1 = self.f1_score(y_hat, y)
        # logging onto tensorboard
        self.log(f"{batch_kind}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{batch_kind}_acc", acc, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log(f"{batch_kind}_f1", f1, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, "test")

    def predict_step(self, batch, batch_idx):
        self.eval()
        x, _, metadata = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        assert self.trainer.max_epochs is not None

        num_iters = self.train_loader_len * self.trainer.max_epochs
        T_M = math.ceil(num_iters / self.num_estimators)

        # cosine annealing that steps at every batch iteration
        def lr_lambda(iteration: int):
            return float(
                0.5 * (torch.cos(torch.tensor(torch.pi * (iteration % T_M) / T_M)) + 1)
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
