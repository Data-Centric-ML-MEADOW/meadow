import argparse
from tqdm import tqdm
import torch
import re
import lightning as L
from utils.mappings import MODEL_MAP, TFMS_MAP
from utils.data import get_iwildcam_datasets, create_loader

L.seed_everything(42)  # for reproducability
torch.set_float32_matmul_precision("high")

EVAL_SPLIT_TYPES = [
    "val",
    "test",
    "id_val",
    "id_test",
]

def parse_checkpoint_filename(checkpoint_path: str):
    """Parse the checkpoint filename to extract model name, variant, and batch size."""
    pattern = r"(?P<model_name>.+)-(?P<model_variant>.+)_(?P<date>\d{8}-\d{6})_lr(?P<lr>[0-9.]+)_bs(?P<batch_size>\d+)\.ckpt"
    match = re.match(pattern, checkpoint_path.split('/')[-1])

    if not match:
        raise ValueError("Checkpoint filename does not match the expected format.")

    return {
        "model_name": match.group("model_name"),
        "model_variant": match.group("model_variant"),
        "lr": float(match.group("lr")),
        "batch_size": int(match.group("batch_size")),
    }

def collect_eval_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint")
    parser.add_argument("--checkpoint-path", required=True, help="Path to the saved model checkpoint file")
    return parser.parse_args()

def main():
    args = collect_eval_arguments()

    # Parse checkpoint file name to get model details
    checkpoint_info = parse_checkpoint_filename(args.checkpoint_path)
    model_name = checkpoint_info["model_name"]
    model_variant = checkpoint_info["model_variant"]
    batch_size = checkpoint_info["batch_size"]

    # Select the model from MODEL_MAP based on the name and variant
    if model_name not in MODEL_MAP:
        raise ValueError(f"Model name must be one of {MODEL_MAP.keys()}")
    model_class = MODEL_MAP[model_name]

    # Set default transformations if none specified
    model_tfms = TFMS_MAP.get(model_name, TFMS_MAP["default"])

    dummy_trainer = L.Trainer()

    # Load the labeled dataset and create evaluation DataLoaders
    labeled_dataset, _ = get_iwildcam_datasets()
    for split in EVAL_SPLIT_TYPES:
        loader = create_loader(
            labeled_dataset,
            subset_type=split,
            tfms=model_tfms,
            batch_size=batch_size
        )
        assert loader is not None

        # Initialize the model from the checkpoint
        model = model_class.load_from_checkpoint(
            args.checkpoint_path,
            resnet_variant=model_variant,
            out_classes=labeled_dataset.n_classes,
        )
        model.eval()

        # infer on the loader
        all_y_pred = dummy_trainer.predict(model=model, dataloaders=loader)
        assert all_y_pred is not None
        # find predictions for each data example
        all_y_pred = torch.vstack(all_y_pred).argmax(dim=-1).flatten()

        # stack all true labels and metadata for the loader
        all_y_true = []
        all_metadata = []
        for _, y_true, metadata in tqdm(loader):
            all_y_true.append(y_true)
            all_metadata.append(metadata)
        all_y_true = torch.hstack(all_y_true)
        all_metadata = torch.vstack(all_metadata)

        # Run evaluation with all accumulated predictions and metadata
        print(f"==={split}===")
        res, _ = labeled_dataset.eval(all_y_pred, all_y_true, all_metadata)
        print(res)

if __name__ == "__main__":
    main()
