import argparse
from datetime import datetime

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.tuner.tuning import Tuner
from torch.utils.data import DataLoader
from torchensemble.utils.logging import set_logger

from utils.data import create_loader, get_iwildcam_datasets
from utils.mappings import ENSEMBLE_MAP, MODEL_MAP, TFMS_MAP

L.seed_everything(42, workers=True)  # for reproducability
torch.set_float32_matmul_precision("high")

MAX_EPOCHS = 50

TIME_NOW_STR = datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S")


def get_trainer(
    log_name: str = "lightning_logs",
    log_save_dir: str = "logs",
    early_stopping_patience: int | None = 5,
) -> L.Trainer:
    """Creates a lightning trainer."""
    # csv and tensorboard loggers for debugging
    loggers = [
        CSVLogger(name=log_name, save_dir=log_save_dir),
        TensorBoardLogger(name=log_name, save_dir=log_save_dir),
    ]

    # configure early stopping if patience is specified
    callbacks = []
    if early_stopping_patience:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss", mode="min", patience=early_stopping_patience
            )
        )
    return L.Trainer(
        logger=loggers, max_epochs=MAX_EPOCHS, callbacks=callbacks, deterministic=True
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
    print(f"Training model w/ description: {run_desc}")

    # fit model and save checkpoint
    trainer = get_trainer(
        early_stopping_patience=args.early_stopping_patience,
        log_save_dir=args.log_save_dir,
        log_name=run_desc,
    )
    trainer.fit(
        model,
        train_dataloaders=labeled_train_loader,
        val_dataloaders=labeled_val_loader,
    )
    ckpt_save_path = f"checkpoints/{run_desc}.ckpt"
    trainer.save_checkpoint(ckpt_save_path)


def train_ensemble(
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
    print(f"Training ensemble model w/ description: {run_desc}")

    set_logger(f"{run_desc}", use_tb_logger=True)

    # if ensemble type is snapshot, then we need to train 10 epochs per estimator
    num_epochs = 10 * (args.num_estimators if args.ensemble_type == "snapshot" else 1)

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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--find-lr", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--log-save-dir", type=str, default="logs")
    parser.add_argument("--ensemble-type", type=str)
    parser.add_argument("--num-estimators", type=int)
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

    # get datasets and dataloaders with transformations
    labeled_dataset, unlabeled_dataset = get_iwildcam_datasets()
    labeled_train_loader = create_loader(
        labeled_dataset,
        "train",
        tfms=model_tfms,
        batch_size=args.batch_size,
        metadata=not args.ensemble_type,  # metadata is not needed for ensembles
    )
    labeled_val_loader = create_loader(
        labeled_dataset,
        "val",
        tfms=val_tfms,
        batch_size=args.batch_size,
        metadata=not args.ensemble_type,
    )
    assert labeled_train_loader is not None
    assert labeled_val_loader is not None

    out_classes = labeled_dataset.n_classes
    assert out_classes is not None

    if not args.ensemble_type:
        train_model(
            out_classes,
            labeled_train_loader,
            labeled_val_loader,
            args,
        )
    else:
        train_ensemble(
            model_class,
            out_classes,
            labeled_train_loader,
            args,
        )
