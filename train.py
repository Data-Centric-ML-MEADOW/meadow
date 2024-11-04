import argparse
from datetime import datetime

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from utils.mappings import MODEL_MAP, TFMS_MAP
from utils.data import get_iwildcam_datasets, create_loader

L.seed_everything(42)  # for reproducability
torch.set_float32_matmul_precision("high")

MAX_EPOCHS = 50

def get_trainer(
    log_name: str = "lightning_logs",
    log_save_dir: str = "logs",
    early_stopping_patience: int | None = 10
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
        logger=loggers,
        max_epochs=MAX_EPOCHS,
        callbacks=callbacks,
    )

def collect_train_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-variant", required=True)
    parser.add_argument("--no-freeze-backbone", action="store_false")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--log-save-dir", type=str, default="logs")
    return parser.parse_args()

if __name__ == "__main__":
    args = collect_train_arguments()

    # create string identifier for model run
    time_str = datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S")
    model_str = f"{args.model_name}-{args.model_variant}"
    run_desc = f"{model_str}_{time_str}_lr{args.lr}_bs{args.batch_size}"
    print(f"Training model w/ description: {run_desc}")

    # select model from mapping based on argument
    if args.model_name not in MODEL_MAP:
        raise ValueError(f"Model name must be one of {MODEL_MAP.keys()}")
    model_class = MODEL_MAP[args.model_name]

    # set a default transform if none is provided
    model_tfms = (
        TFMS_MAP[args.model_name]
        if args.model_name in TFMS_MAP
        else TFMS_MAP["default"]
    )

    # get datasets and dataloaders with transformations
    labeled_dataset, unlabeled_dataset = get_iwildcam_datasets()
    labeled_train_loader = create_loader(
        labeled_dataset,
        "train",
        tfms=model_tfms,
        batch_size=args.batch_size
    )
    labeled_val_loader = create_loader(
        labeled_dataset,
        "val",
        tfms=model_tfms,
        batch_size=args.batch_size
    )

    # create model instance
    model = model_class(
        resnet_variant=args.model_variant,
        out_classes=labeled_dataset.n_classes,
        lr=args.lr,
        freeze_backbone=args.no_freeze_backbone,
    )

    # fit model and save checkpoint
    trainer = get_trainer(
        early_stopping_patience=args.early_stopping_patience,
        log_save_dir=args.log_save_dir,
        log_name=run_desc
    )
    trainer.fit(
        model,
        train_dataloaders=labeled_train_loader,
        val_dataloaders=labeled_val_loader
    )
    ckpt_save_path = f"checkpoints/{run_desc}.ckpt"
    trainer.save_checkpoint(ckpt_save_path)
