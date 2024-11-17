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
        domain_mapper: torch.nn.Module,
        out_classes,
        learn_domain_mapper: bool = False,
        optimizer=torch.optim.AdamW,  # type: ignore
        lr=1e-3,
        snap_router_on_epoch: int | None = None,
        learn_experts_after: int = -1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_domains = num_domains
        self.out_classes = out_classes

        self.expert_models = expert_models
        # ensure all experts have frozen backbones
        for expert in self.expert_models:
            expert.freeze_backbone = True  # type: ignore
        self.num_experts = len(self.expert_models)

        self.domain_mapper = domain_mapper
        self.learn_domain_mapper = learn_domain_mapper
        self.snap_router_on_epoch = snap_router_on_epoch
        self.learn_experts_after = learn_experts_after

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
        # keep the domain mapper in eval mode if specified
        if not self.learn_domain_mapper:
            self.domain_mapper.eval()

    def eval(self):
        super().eval()
        for expert in self.expert_models:
            expert.eval()
        self.domain_mapper.eval()

    def forward(self, x, d):
        # router should return a vector of weights for each expert
        expert_weights = F.softmax(self.router(d), dim=-1)
        model_outputs = torch.stack([m(x) for m in self.expert_models])
        # (E x B x C) * (E x B x 1) -- broadcasting across classes
        return (model_outputs * expert_weights.T[:, :, None]).sum(dim=0)

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        if (
            self.snap_router_on_epoch
            and self.current_epoch == self.snap_router_on_epoch
        ):
            # perform snapping on the router on `snap_router_on_epoch`
            with torch.no_grad():
                self.router.weight = torch.nn.Parameter(
                    F.one_hot(
                        torch.argmax(self.router.weight, dim=0),
                        num_classes=self.num_experts,
                    ).T.float(),
                    requires_grad=False,
                )

    def training_step(self, batch, batch_idx):
        self.train()
        x, y, _ = batch
        d = self.domain_mapper(x)
        y_hat = self(x, d)

        # init optimizers for train step
        router_opt, expert_opt, domain_opt = self.optimizers()  # type: ignore
        if (
            not self.snap_router_on_epoch
            or self.current_epoch < self.snap_router_on_epoch
        ):
            router_opt.zero_grad()  # type: ignore
        if self.current_epoch > self.learn_experts_after:
            expert_opt.zero_grad()  # type: ignore
        if self.learn_domain_mapper:
            domain_opt.zero_grad()  # type: ignore

        # calculate loss and call backward
        loss = F.cross_entropy(y_hat, y)
        self.manual_backward(loss)

        # step optimizers
        if (
            not self.snap_router_on_epoch
            or self.current_epoch < self.snap_router_on_epoch
        ):
            router_opt.step()  # type: ignore
        # only start optimizing the experts after a few epochs
        if self.current_epoch > self.learn_experts_after:
            expert_opt.step()  # type: ignore
        if self.learn_domain_mapper:
            domain_opt.step()  # type: ignore

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
        self.eval()
        x, y, _ = batch
        with torch.no_grad():
            d = self.domain_mapper(x)
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
            d = self.domain_mapper(x)
        return self(x, d)

    def configure_optimizers(self):
        # have router learn steeper than expert and domain mapper
        router_opt = self.optimizer(self.router.parameters(), lr=self.lr)
        expert_opt = self.optimizer(self.expert_models.parameters(), lr=self.lr)
        domain_opt = self.optimizer(self.domain_mapper.parameters(), lr=self.lr)
        return router_opt, expert_opt, domain_opt
