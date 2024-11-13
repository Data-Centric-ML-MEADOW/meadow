import argparse
from datetime import datetime

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.tuner.tuning import Tuner
from torch.utils.data import DataLoader
from torchensemble.utils.logging import set_logger

from models.snapshot_ensemble import SnapshotEnsemble
from utils.data import create_loader, get_iwildcam_datasets
from utils.mappings import ENSEMBLE_MAP, MODEL_MAP, TFMS_MAP

L.seed_everything(42, workers=True)  # for reproducability
torch.set_float32_matmul_precision("high")

MAX_EPOCHS = 50

TIME_NOW_STR = datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S")


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


def find_lr(model: L.LightningModule, train_loader: DataLoader) -> float | None:
    """Finds the optimal learning rate for a given model and dataloader."""
    # create a dummy trainer to find learning rate with tuner
    trainer = L.Trainer(max_epochs=MAX_EPOCHS, deterministic=True, logger=[])
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, train_loader)
    if not lr_finder:
        return None
    return lr_finder.suggestion()


def train_model(
    model_class,
    out_classes: int,
    labeled_train_loader: DataLoader,
    labeled_val_loader: DataLoader,
    args: argparse.Namespace,
):
    """Trains a regular model with the given dataloaders."""
    # create model instance
    model = model_class(
        variant=args.model_variant,
        out_classes=out_classes,
        lr=args.lr,
        freeze_backbone=args.no_freeze_backbone,
    )

    # if specified, find an optimal learning rate to use instead
    if args.find_lr:
        optimal_lr = find_lr(model, labeled_train_loader)
        if optimal_lr is not None:
            print(f"Found optimal LR: {optimal_lr}")
            model.hparams.lr = optimal_lr
            model.lr = optimal_lr
            args.lr = optimal_lr
        else:
            print("No optimal LR found, using set LR")

    # create string identifier for model run
    model_str = f"{args.model_name}-{args.model_variant}"
    run_desc = f"{model_str}_{TIME_NOW_STR}_lr{args.lr:.2e}_bs{args.batch_size}"
    if args.misc_desc:
        run_desc += f"[{args.misc_desc}]"
    print(f"Training model w/ description: {run_desc}")

    # fit model and save checkpoint
    trainer = get_trainer(
        early_stopping_patience=args.early_stopping_patience,
        log_save_dir=args.log_save_dir,
        log_name=run_desc,
        num_epochs=args.epochs,
    )
    trainer.fit(
        model,
        train_dataloaders=labeled_train_loader,
        val_dataloaders=labeled_val_loader,
    )
    ckpt_save_path = f"checkpoints/{run_desc}.ckpt"
    trainer.save_checkpoint(ckpt_save_path)


def train_ensemble_pl(
    base_model_class,
    out_classes: int,
    labeled_train_loader: DataLoader,
    labeled_val_loader: DataLoader,
    args: argparse.Namespace,
):
    """Trains an ensemble of models with the given dataloaders."""
    model_args = {
        "variant": args.model_variant,
        "out_classes": out_classes,
        "lr": args.lr,
        "freeze_backbone": args.no_freeze_backbone,
    }

    # create string identifier for model run
    model_str = f"{args.model_name}-{args.model_variant}"
    run_desc = (
        f"{model_str}_{TIME_NOW_STR}_lr{args.lr:.2e}_bs{args.batch_size}"
        f"_{args.ensemble_type}{args.num_estimators}"
    )
    if args.misc_desc:
        run_desc += f"[{args.misc_desc}]"
    print(f"Training model w/ description: {run_desc}")

    ensemble_model = SnapshotEnsemble(
        out_classes=out_classes,
        train_loader_len=len(labeled_train_loader),
        base_model=base_model_class,
        base_model_args=model_args,
        num_estimators=args.num_estimators,
    )

    # fit model and save checkpoint
    trainer = get_trainer(
        log_save_dir=args.log_save_dir,
        log_name=run_desc,
        num_epochs=args.epochs * args.num_estimators,  # num epochs per estimator
        lr_monitor=True,
        early_stopping_patience=None,  # train all estimators
        strategy=(
            "ddp_find_unused_parameters_true"
            if torch.cuda.device_count() > 1
            else "auto"
        ),
    )
    trainer.fit(
        ensemble_model,
        train_dataloaders=labeled_train_loader,
        val_dataloaders=labeled_val_loader,
    )
    ckpt_save_path = f"checkpoints/{run_desc}.ckpt"
    trainer.save_checkpoint(ckpt_save_path)


def train_torchensemble(
    base_model_class,
    out_classes: int,
    labeled_train_loader: DataLoader,
    args: argparse.Namespace,
):
    # select ensemble type from mapping based on argument
    try:
        ensemble_class = ENSEMBLE_MAP[args.ensemble_type]
    except KeyError:
        raise ValueError(f"Ensemble type must be one of {ENSEMBLE_MAP.keys()}")

    if args.num_estimators is None:
        raise ValueError("Number of estimators must be specified")

    # create ensemble model
    ensemble_model = ensemble_class(
        estimator=base_model_class,
        estimator_args={
            "out_classes": out_classes,
            "variant": args.model_variant,
            "freeze_backbone": args.no_freeze_backbone,
        },
        n_estimators=args.num_estimators,
        cuda=True,
    )
    # set cross entropy loss criterion
    criterion = torch.nn.CrossEntropyLoss()
    ensemble_model.set_criterion(criterion)
    # set AdamW optimizer
    ensemble_model.set_optimizer("AdamW", lr=args.lr)

    model_str = f"{args.model_name}-{args.model_variant}"
    run_desc = (
        f"{model_str}_{TIME_NOW_STR}_lr{args.lr:.2e}_bs{args.batch_size}"
        f"_{args.ensemble_type}{args.num_estimators}"
    )
    if args.misc_desc:
        run_desc += f"[{args.misc_desc}]"
    print(f"Training ensemble model w/ description: {run_desc}")

    set_logger(f"{run_desc}")

    # if ensemble type is snapshot, then we need to train 10 epochs per estimator
    num_epochs = args.epochs * (
        args.num_estimators if args.ensemble_type == "snapshot" else 1
    )

    ensemble_model.fit(
        labeled_train_loader,
        epochs=num_epochs,
        # test_loader=labeled_val_loader,
        # do not use a test_loader, it removes the guarantee of saving later estimators
        save_model=True,
        save_dir=f"checkpoints/{run_desc}",
    )


def collect_train_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-variant", required=True)
    parser.add_argument("--no-freeze-backbone", action="store_false")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--find-lr", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--log-save-dir", type=str, default="logs")
    parser.add_argument("--ensemble-type", type=str)
    parser.add_argument("--num-estimators", type=int)
    parser.add_argument("--misc-desc", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = collect_train_arguments()

    using_tfms = args.model_name.endswith("-tfms")

    model_name = args.model_name
    model_name_base = model_name if not using_tfms else model_name.replace("-tfms", "")

    # select model from mapping based on argument
    try:
        model_class = MODEL_MAP[model_name_base]
    except KeyError:
        raise ValueError(f"Model name must be one of {MODEL_MAP.keys()}")

    try:
        # set a default transform if none is provided
        model_tfms = TFMS_MAP.get(model_name, TFMS_MAP["default"])
        # for validation transforms, do not use image augmentations
        val_tfms = TFMS_MAP.get(model_name_base, TFMS_MAP["default"])
    except KeyError as e:
        raise ValueError(f"Transformations for model '{model_name}' not found: {e}")

    using_torchensemble = args.ensemble_type and args.ensemble_type != "snapshot-pl"

    # get datasets and dataloaders with transformations
    labeled_dataset, unlabeled_dataset = get_iwildcam_datasets()
    labeled_train_loader = create_loader(
        labeled_dataset,
        "train",
        tfms=model_tfms,
        batch_size=args.batch_size,
        # metadata is not needed for torchensemble ensembles
        metadata=not using_torchensemble,
    )
    labeled_val_loader = create_loader(
        labeled_dataset,
        "val",
        tfms=val_tfms,
        batch_size=args.batch_size,
        metadata=not using_torchensemble,
    )
    assert labeled_train_loader is not None
    assert labeled_val_loader is not None

    out_classes = labeled_dataset.n_classes
    assert out_classes is not None

    if not args.ensemble_type:
        train_model(
            model_class,
            out_classes,
            labeled_train_loader,
            labeled_val_loader,
            args,
        )
    elif args.ensemble_type == "snapshot-pl":
        train_ensemble_pl(
            model_class,
            out_classes,
            labeled_train_loader,
            labeled_val_loader,
            args,
        )
    else:
        train_torchensemble(
            model_class,
            out_classes,
            labeled_train_loader,
            args,
        )
