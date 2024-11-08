import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, MulticlassF1Score


class DomainMoE(L.LightningModule):
    def __init__(
        self,
        num_domains,
        expert_model_class,
        num_experts,
        out_classes,
        optimizer=torch.optim.AdamW,  # type: ignore
        lr=1e-4,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.expert_model_class = expert_model_class
        self.num_experts = num_experts
        self.out_classes = out_classes

        self.optimizer = optimizer
        self.lr = lr

        # accuracy metric for train/val loop
        self.accuracy = Accuracy(task="multiclass", num_classes=self.out_classes)
        self.f1_score = MulticlassF1Score(num_classes=self.out_classes, average="macro")

        # router to select expert model from set of domains
        self.router = torch.nn.Linear(self.num_domains, self.num_experts)
        # initialize multiple expert models
        self.expert_models = torch.nn.ModuleList(
            [self.expert_model_class(**kwargs) for _ in range(self.num_experts)]
        )
        # concatenate components together for the MoE model
        self.model = torch.nn.Sequential(self.router, self.expert_models)

    def forward(self, x):
        return self.model(x)

    def _batch_step(self, batch, batch_kind):
        if batch_kind == "train":
            self.model.train()
        else:
            self.model.eval()
        x, y, metadata = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        f1 = self.f1_score(y_hat, y)
        # logging
        self.log(f"{batch_kind}_loss", loss, prog_bar=True)
        self.log(f"{batch_kind}_acc", acc, prog_bar=True, on_epoch=True)
        self.log(f"{batch_kind}_f1", f1, prog_bar=True, on_epoch=True)
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
