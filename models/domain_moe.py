import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, MulticlassF1Score


class DomainMoE(L.LightningModule):
    """Domain-aware mixture of experts model. Accepts a module list, with each element an expert instance."""

    def __init__(
        self,
        num_domains,
        expert_models: torch.nn.ModuleList,
        test_domain_mapper: torch.nn.Module,
        out_classes,
        optimizer=torch.optim.AdamW,  # type: ignore
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_domains = num_domains
        self.num_experts = len(expert_models)
        self.out_classes = out_classes
        self.expert_models = expert_models
        self.test_domain_mapper = test_domain_mapper

        self.optimizer = optimizer
        self.lr = lr

        # accuracy metric for train/val loop
        self.accuracy = Accuracy(task="multiclass", num_classes=self.out_classes)
        self.f1_score = MulticlassF1Score(num_classes=self.out_classes, average="macro")

        # router to select expert model from set of domains
        self.router = torch.nn.Linear(self.num_domains, self.num_experts)

        # disable automatic optimization to allow for different lr
        self.automatic_optimization = False

    def train(self, mode=True):
        super().train(mode)
        for expert in self.expert_models:
            expert.train(mode)
        # keep the domain mapper in eval mode
        self.test_domain_mapper.eval()

    def eval(self):
        super().eval()
        for expert in self.expert_models:
            expert.eval()

    def forward(self, x, d):
        expert_weights = self.router(d)
        torch.stack(
            [m(x) * expert_weights[:, i] for i, m in enumerate(self.expert_models)],
            dim=0,
        ).mean(dim=0)

    def training_step(self, batch, batch_idx):
        self.train()
        x, y, metadata = batch
        d = F.one_hot(metadata[:, 0], num_classes=self.num_domains)
        y_hat = self(x, d)

        router_opt, expert_opt = self.optimizers()  # type: ignore
        router_opt.zero_grad()  # type: ignore
        expert_opt.zero_grad()  # type: ignore
        loss = F.cross_entropy(y_hat, y)
        self.manual_backward(loss)
        router_opt.zero_grad()  # type: ignore
        expert_opt.zero_grad()  # type: ignore

        acc = self.accuracy(y_hat, y)
        f1 = self.f1_score(y_hat, y)
        # logging onto tensorboard
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log(
            "train_acc",
            acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train_f1", f1, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )
        return loss

    def _eval_step(self, batch, batch_kind):
        if batch_kind == "train":
            self.train()
        else:
            self.eval()
        x, y, _ = batch
        with torch.no_grad():
            d = self.test_domain_mapper(x)
        y_hat = self(x, d)
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        f1 = self.f1_score(y_hat, y)
        # logging
        self.log(f"{batch_kind}_loss", loss, prog_bar=True)
        self.log(f"{batch_kind}_acc", acc, prog_bar=True, on_epoch=True)
        self.log(f"{batch_kind}_f1", f1, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, "test")

    def predict_step(self, batch, batch_idx):
        # assume on predict we are evaluating against OOD data
        self.eval()
        x, _, _ = batch
        with torch.no_grad():
            d = self.test_domain_mapper(x)
        return self(x, d)

    def configure_optimizers(self):
        # have router learn steeper than expert
        router_opt = self.optimizer(self.router.parameters(), lr=self.lr)
        expert_opt = self.optimizer(self.expert_models.parameters(), lr=self.lr * 1e-2)
        return router_opt, expert_opt
        # return self.optimizer(self.parameters(), lr=self.lr)
