import ssl
import typing
from datetime import datetime

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torchvision import transforms
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader, get_train_loader
from wilds.datasets.iwildcam_dataset import IWildCamDataset
from wilds.datasets.unlabeled.iwildcam_unlabeled_dataset import IWildCamUnlabeledDataset

from models.base_pretrained import PreTrainedResNet

ssl._create_default_https_context = ssl._create_unverified_context
L.seed_everything(42)  # for reproducability
torch.set_float32_matmul_precision("high")

BATCH_SIZE = 16
MAX_EPOCHS = 80


def get_iwildcam_datasets() -> tuple[IWildCamDataset, IWildCamUnlabeledDataset]:
    """Retrieves the iWildCam labeled and unlabeled datasets."""
    dataset = typing.cast(  # explicit cast to make type checker happy
        IWildCamDataset | None,
        get_dataset(dataset="iwildcam", download=True),
    )
    dataset_unlabeled = typing.cast(
        IWildCamUnlabeledDataset | None,
        get_dataset(dataset="iwildcam", download=True, unlabeled=True),
    )
    # error out if dataset not found
    if not dataset or not dataset_unlabeled:
        raise RuntimeError("Dataset not found, or failed to download.")
    return dataset, dataset_unlabeled


def create_loader(
    dataset: IWildCamDataset | IWildCamUnlabeledDataset,
    subset_type: str,
    tfms: transforms.Compose = transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor()]
    ),
    batch_size=BATCH_SIZE,
    num_workers=4,
):
    """Creates a train dataloader."""
    if subset_type not in dataset.split_dict:
        raise ValueError(f"subset_type must be one of {dataset.split_dict.keys()}")

    data = dataset.get_subset(split=subset_type, transform=tfms)
    loader = get_train_loader if subset_type == "train" else get_eval_loader

    return loader(
        loader="standard",
        dataset=data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_trainer() -> L.Trainer:
    """Creates a lightning trainer."""
    # csv and tensorboard loggers for debugging
    loggers = [
        CSVLogger(save_dir="logs"),
        TensorBoardLogger(save_dir="logs"),
    ]
    # don't overfit
    early_stoper = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    return L.Trainer(
        logger=loggers,
        max_epochs=MAX_EPOCHS,
        callbacks=[early_stoper],
    )


if __name__ == "__main__":
    # get datasets and dataloaders
    labeled_dataset, unlabeled_dataset = get_iwildcam_datasets()
    resnet_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    labeled_train_loader = create_loader(
        labeled_dataset, "train", tfms=resnet_transforms
    )
    labeled_val_loader = create_loader(labeled_dataset, "val", tfms=resnet_transforms)
    trainer = get_trainer()

    # init model
    resnet_variant = 101
    model = PreTrainedResNet(
        resnet_variant=resnet_variant, out_classes=labeled_dataset.n_classes
    )

    # fit model and save checkpoint
    trainer.fit(model, labeled_train_loader, labeled_val_loader)
    trainer.save_checkpoint(f"checkpoints/resnet{resnet_variant}_{datetime.now()}.ckpt")
