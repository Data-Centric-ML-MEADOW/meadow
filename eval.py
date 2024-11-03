import argparse
from tqdm import tqdm
import torch
import re
import lightning as L
from utils.mappings import MODEL_MAP, TFMS_MAP
from utils.data import get_iwildcam_datasets, create_loader

def parse_checkpoint_filename(checkpoint_path: str):
    """Parse the checkpoint filename to extract model name, variant, and batch size."""
    pattern = r"(?P<model_name>.+)-(?P<model_variant>.+)_(?P<date>\d{8}_\d{6})_lr(?P<lr>[0-9.]+)_bs(?P<batch_size>\d+)\.ckpt"
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

    # Load the validation dataset and create a DataLoader
    labeled_dataset, _ = get_iwildcam_datasets()
    labeled_val_loader = create_loader(
        labeled_dataset,
        "val",
        tfms=model_tfms,
        batch_size=batch_size
    )

    # Initialize the model from the checkpoint
    model = model_class.load_from_checkpoint(
        args.checkpoint_path,
        resnet_variant=model_variant,
        out_classes=labeled_dataset.n_classes,
    )

    model.eval()

    # Assuming you already have these lists defined
    all_preds = []
    all_y_true = []
    all_metadata = []

    # Wrap labeled_val_loader with tqdm to display a progress bar
    for x, y, metadata in tqdm(labeled_val_loader, desc="Evaluating", total=len(labeled_val_loader)):
        y_pred = model(x)
        all_preds.append(y_pred)
        all_y_true.append(y)
        all_metadata.append(metadata)
        break

    # Stack predictions, true labels, and metadata
    all_preds = torch.hstack(all_preds)
    all_y_true = torch.hstack(all_y_true)
    all_metadata = torch.hstack(all_metadata)

    # Run evaluation with all accumulated predictions and metadata
    labeled_dataset.eval(all_preds, all_y_true, all_metadata)


if __name__ == "__main__":
    main()
