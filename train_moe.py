import argparse
import re
from datetime import datetime

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from models.domain_moe import DomainMoE
from models.snapshot_ensemble import SnapshotEnsemble
from utils.data import create_loader, get_iwildcam_datasets
from utils.mappings import MODEL_MAP, TFMS_MAP

L.seed_everything(42, workers=True)
torch.set_float32_matmul_precision("high")

MAX_EPOCHS = 50

TIME_NOW_STR = datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S")


def parse_checkpoint_filename(checkpoint_path: str):
    """Parse the checkpoint filename to extract model name, variant, and batch size."""
    pattern = r"(?P<model_name>.+)-(?P<model_variant>.+)_(?P<date>\d{8}-\d{6})_lr(?P<lr>[e0-9.-]+)_bs(?P<batch_size>\d+)(?:_(?P<ensemble>[a-z-]+)(?P<n_estimators>\d+))?(?:\[.+\])?\.ckpt"
    match = re.match(pattern, checkpoint_path.split("/")[-1])

    if not match:
        raise ValueError("Checkpoint filename does not match the expected format.")

    return {
        "model_name": match.group("model_name"),
        "model_variant": match.group("model_variant"),
        "lr": float(match.group("lr")),
        "batch_size": int(match.group("batch_size")),
        "ensemble": match.group("ensemble"),
        "n_estimators": int(match.group("n_estimators") or -1),
    }


def get_trainer(
    num_epochs: int = MAX_EPOCHS,
    log_name: str = "lightning_logs",
    log_save_dir: str = "logs",
    early_stopping_patience: int | None = 5,
    lr_monitor: bool = False,
    **kwargs,
) -> L.Trainer:
    """Creates a lightning trainer."""
    # csv and tensorboard loggers for debugging
    loggers = [
        CSVLogger(name=log_name, save_dir=log_save_dir),
        TensorBoardLogger(name=log_name, save_dir=log_save_dir),
    ]

    # configure early stopping if patience is specified
    callbacks = []
    if lr_monitor:
        callbacks.append(LearningRateMonitor("step"))
    if early_stopping_patience:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss", mode="min", patience=early_stopping_patience
            )
        )
    return L.Trainer(
        logger=loggers,
        max_epochs=num_epochs,
        callbacks=callbacks,
        deterministic=True,
        **kwargs,
    )


def get_experts_from_ensemble(
    expert_ensemble_checkpoint_path: str, out_classes: int
) -> tuple[torch.nn.ModuleList, dict]:
    checkpoint_info = parse_checkpoint_filename(expert_ensemble_checkpoint_path)
    expert_model_name = checkpoint_info["model_name"]
    expert_model_name_base = expert_model_name.split("-")[0]
    expert_model_variant = checkpoint_info["model_variant"]

    expert_model_class = MODEL_MAP.get(
        expert_model_name, MODEL_MAP[expert_model_name_base]
    )

    # extract the experts from the learned ensemble
    # TODO: add support for torchensemble
    ensemble_model = SnapshotEnsemble.load_from_checkpoint(
        checkpoint_path=args.expert_ensemble_checkpoint_path,
        out_classes=out_classes,
        train_loader_len=-1,
        num_estimators=checkpoint_info["n_estimators"],
        base_model=expert_model_class,
        base_model_args={"variant": expert_model_variant, "out_classes": out_classes},
    )
    return (ensemble_model.ensemble, checkpoint_info)


def get_domain_mapper_model(
    domain_mapper_checkpoint_path: str, out_classes: int
) -> tuple[torch.nn.Module, dict]:
    checkpoint_info = parse_checkpoint_filename(domain_mapper_checkpoint_path)
    domain_mapper_model_name = checkpoint_info["model_name"]
    domain_mapper_model_variant = checkpoint_info["model_variant"]
    domain_mapper_model_class = MODEL_MAP[domain_mapper_model_name]
    return (
        domain_mapper_model_class.load_from_checkpoint(
            domain_mapper_checkpoint_path,
            variant=domain_mapper_model_variant,
            out_classes=out_classes,
        ),
        checkpoint_info,
    )


def collect_train_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-ensemble-checkpoint-path", required=True)
    parser.add_argument("--domain-mapper-checkpoint-path", required=True)
    # hyperparams for the MoE
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--learn-domain-mapper", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--log-save-dir", type=str, default="logs")
    # descriptive string to append to log/checkpoint files (e.g. experiment setup, num gpus, etc.)
    parser.add_argument("--misc-desc", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = collect_train_arguments()

    # Load the labeled dataset
    labeled_dataset, _ = get_iwildcam_datasets()
    out_classes = labeled_dataset.n_classes
    assert out_classes is not None
    num_labeled_domains = len(torch.unique(labeled_dataset.metadata_array[:, 0]))

    # get the experts, transforms, and domain mapper
    experts, expert_checkpoint_info = get_experts_from_ensemble(
        args.expert_ensemble_checkpoint_path, out_classes
    )

    domain_mapper, domain_mapper_checkpoint_info = get_domain_mapper_model(
        args.domain_mapper_checkpoint_path, num_labeled_domains
    )

    tfms = TFMS_MAP[expert_checkpoint_info["model_name"]]

    # create dataloaders
    labeled_train_loader = create_loader(
        labeled_dataset,
        "train",
        tfms=tfms,
        batch_size=args.batch_size,
    )
    labeled_val_loader = create_loader(
        labeled_dataset,
        "val",
        tfms=tfms,
        batch_size=args.batch_size,
    )
    assert labeled_train_loader is not None
    assert labeled_val_loader is not None

    # create the MoE model
    moe_model = DomainMoE(
        num_domains=num_labeled_domains,
        expert_models=experts,
        domain_mapper=domain_mapper,
        out_classes=out_classes,
        lr=args.lr,
        learn_domain_mapper=args.learn_domain_mapper,
    )

    run_desc = (
        f"MoE{moe_model.num_experts}"
        f"_{expert_checkpoint_info['model_name']}-{expert_checkpoint_info['model_variant']}"
        f"_{domain_mapper_checkpoint_info['model_name']}-{domain_mapper_checkpoint_info['model_variant']}"
        f"_{TIME_NOW_STR}_lr{args.lr:.2e}_bs{args.batch_size}"
    )
    if args.misc_desc:
        run_desc += f"[{args.misc_desc}]"
    print(f"Training MoE w/ description: {run_desc}")

    # retrieve trainer
    trainer = get_trainer(
        log_save_dir=args.log_save_dir,
        log_name=run_desc,
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
    )

    # fit model and save checkpoint
    trainer.fit(
        moe_model,
        train_dataloaders=labeled_train_loader,
        val_dataloaders=labeled_val_loader,
    )
    ckpt_save_path = f"checkpoints/{run_desc}.ckpt"
    trainer.save_checkpoint(ckpt_save_path)
