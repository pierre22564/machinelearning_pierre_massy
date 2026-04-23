from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


NUM_CLASSES = 4
DATASET_MEAN = (0.27182294, 0.04167788, 0.36513724)
DATASET_STD = (0.01782968, 0.05269813, 0.03825243)


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


class RoomOccupancyDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        ids: Iterable[int],
        targets: Iterable[int] | None = None,
        transform: transforms.Compose | None = None,
        input_mode: str = "rgb",
    ) -> None:
        self.image_dir = Path(image_dir)
        self.ids = list(ids)
        self.targets = None if targets is None else list(targets)
        self.transform = transform
        self.input_mode = input_mode

    def __len__(self) -> int:
        return len(self.ids)

    def _open_image(self, sample_id: int) -> Image.Image:
        image_path = self.image_dir / f"img_{sample_id + 1}.png"
        image = Image.open(image_path)
        if self.input_mode == "gray3":
            gray = image.convert("L")
            image = Image.merge("RGB", (gray, gray, gray))
        else:
            image = image.convert("RGB")
        return image

    def __getitem__(self, idx: int):
        sample_id = self.ids[idx]
        image = self._open_image(sample_id)
        if self.transform:
            image = self.transform(image)
        if self.targets is None:
            return image, sample_id
        target = int(self.targets[idx])
        return image, target, sample_id


class TinyRoomCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, drop_rate: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    ops = [transforms.Resize((image_size, image_size), antialias=True)]
    if train:
        ops.extend(
            [
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        ]
    )
    if train:
        ops.append(transforms.RandomErasing(p=0.15, scale=(0.01, 0.08), ratio=(0.3, 3.0)))
    return transforms.Compose(ops)


def create_model(model_name: str, num_classes: int, drop_rate: float) -> nn.Module:
    if model_name == "tinycnn":
        return TinyRoomCNN(num_classes=num_classes, drop_rate=drop_rate)
    return timm.create_model(model_name, pretrained=True, num_classes=num_classes, drop_rate=drop_rate)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    sampler: WeightedRandomSampler | None = None,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
    )


def build_sampler(targets: np.ndarray) -> WeightedRandomSampler:
    counts = Counter(targets.tolist())
    weights = np.array([1.0 / counts[int(target)] for target in targets], dtype=np.float64)
    return WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, targets, _sample_ids in loader:
        images = images.to(device)
        targets = targets.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, targets)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_seen += images.size(0)

    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_ids = []
    for batch in loader:
        if len(batch) == 2:
            images, sample_ids = batch
        else:
            images, _targets, sample_ids = batch
        images = images.to(device)
        probs = torch.softmax(model(images), dim=1).cpu().numpy()
        all_probs.append(probs)
        if torch.is_tensor(sample_ids):
            sample_ids = sample_ids.cpu().numpy()
        all_ids.append(np.asarray(sample_ids, dtype=np.int64))
    return np.concatenate(all_probs), np.concatenate(all_ids)


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_confusion(cm: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def plot_history(history: list[EpochMetrics], path: Path, title: str) -> None:
    epochs = [m.epoch for m in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [m.train_acc for m in history], label="train_acc")
    plt.plot(epochs, [m.val_acc for m in history], label="val_acc")
    plt.plot(epochs, [m.train_loss for m in history], label="train_loss")
    plt.plot(epochs, [m.val_loss for m in history], label="val_loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def parse_experiment(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        return spec, "rgb"
    model_name, input_mode = spec.split(":", 1)
    return model_name, input_mode


def fit_single_experiment(
    experiment_name: str,
    model_name: str,
    input_mode: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_dir: Path,
    test_dir: Path,
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    id_to_position = {int(sample_id): pos for pos, sample_id in enumerate(train_df["id"].tolist())}

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_probs = np.zeros((len(train_df), NUM_CLASSES), dtype=np.float32)
    test_probs = np.zeros((len(test_df), NUM_CLASSES), dtype=np.float32)
    histories: list[dict] = []
    fold_scores: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(train_df["id"], train_df["target"]), start=1):
        fold_dir = experiment_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_train = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val = train_df.iloc[val_idx].reset_index(drop=True)

        train_ds = RoomOccupancyDataset(
            image_dir=train_dir,
            ids=fold_train["id"].tolist(),
            targets=fold_train["target"].tolist(),
            transform=build_transforms(args.image_size, train=True),
            input_mode=input_mode,
        )
        val_ds = RoomOccupancyDataset(
            image_dir=train_dir,
            ids=fold_val["id"].tolist(),
            targets=fold_val["target"].tolist(),
            transform=build_transforms(args.image_size, train=False),
            input_mode=input_mode,
        )
        test_ds = RoomOccupancyDataset(
            image_dir=test_dir,
            ids=test_df["id"].tolist(),
            targets=None,
            transform=build_transforms(args.image_size, train=False),
            input_mode=input_mode,
        )

        sampler = build_sampler(fold_train["target"].to_numpy()) if args.balance_sampler else None
        train_loader = make_loader(train_ds, args.batch_size, shuffle=True, sampler=sampler, num_workers=args.num_workers)
        val_loader = make_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = make_loader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = create_model(model_name, num_classes=NUM_CLASSES, drop_rate=args.drop_rate).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        best_val_acc = -math.inf
        best_epoch = 0
        best_state = None
        patience_left = args.patience
        fold_history: list[EpochMetrics] = []

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None, device=device)
            scheduler.step()

            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=float(train_loss),
                train_acc=float(train_acc),
                val_loss=float(val_loss),
                val_acc=float(val_acc),
            )
            fold_history.append(metrics)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = args.patience
            else:
                patience_left -= 1

            print(
                f"[{experiment_name}] fold={fold} epoch={epoch} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
                flush=True,
            )

            if patience_left <= 0:
                break

        if best_state is None:
            raise RuntimeError(f"No checkpoint captured for {experiment_name} fold {fold}")

        model.load_state_dict(best_state)
        checkpoint_path = fold_dir / "best_model.pt"
        torch.save(
            {
                "experiment": experiment_name,
                "model_name": model_name,
                "input_mode": input_mode,
                "fold": fold,
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "state_dict": best_state,
            },
            checkpoint_path,
        )

        val_probs, val_ids = predict_probs(model, val_loader, device)
        test_fold_probs, test_ids = predict_probs(model, test_loader, device)
        if not np.array_equal(test_ids, test_df["id"].to_numpy()):
            raise RuntimeError("Test ids changed during inference.")

        for probs, sample_id in zip(val_probs, val_ids):
            oof_probs[id_to_position[int(sample_id)]] = probs
        test_probs += test_fold_probs / args.folds

        val_preds = val_probs.argmax(axis=1)
        val_true = fold_val.set_index("id").loc[val_ids, "target"].to_numpy()
        fold_acc = accuracy_score(val_true, val_preds)
        fold_scores.append(float(fold_acc))

        plot_history(fold_history, fold_dir / "history.png", f"{experiment_name} fold {fold}")
        histories.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "history": [asdict(metric) for metric in fold_history],
            }
        )
        print(f"[{experiment_name}] fold={fold} best_val_acc={fold_acc:.4f} best_epoch={best_epoch}", flush=True)

    y_true = train_df["target"].to_numpy()
    oof_preds = oof_probs.argmax(axis=1)
    oof_acc = accuracy_score(y_true, oof_preds)
    cm = confusion_matrix(y_true, oof_preds)
    plot_confusion(
        cm,
        labels=["0 person", "1 person", "2 persons", "3 persons"],
        path=experiment_dir / "confusion_matrix.png",
        title=f"{experiment_name} OOF confusion matrix",
    )

    np.save(experiment_dir / "oof_probs.npy", oof_probs)
    np.save(experiment_dir / "test_probs.npy", test_probs)

    summary = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "input_mode": input_mode,
        "fold_scores": fold_scores,
        "oof_accuracy": float(oof_acc),
        "histories": histories,
    }
    save_json(summary, experiment_dir / "summary.json")
    return summary


def build_class_distribution_plot(train_df: pd.DataFrame, path: Path) -> None:
    counts = train_df["target"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", color="#336699")
    plt.xticks(rotation=0)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Training class distribution")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-validated training for room occupancy classification.")
    parser.add_argument("--data-root", type=Path, required=True, help="Root directory containing train_images and test_images.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", default=["tinycnn:rgb", "tinycnn:gray3"])
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--drop-rate", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--balance-sampler", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = choose_device()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv).sort_values("id").reset_index(drop=True)
    test_df = pd.read_csv(args.sample_submission).sort_values("id").reset_index(drop=True)
    train_dir = args.data_root / "train_images"
    test_dir = args.data_root / "test_images"

    build_class_distribution_plot(train_df, output_dir / "class_distribution.png")

    summaries = []
    experiment_weights = []
    for spec in args.experiments:
        model_name, input_mode = parse_experiment(spec)
        experiment_name = f"{model_name.replace('/', '_')}__{input_mode}"
        summary = fit_single_experiment(
            experiment_name=experiment_name,
            model_name=model_name,
            input_mode=input_mode,
            train_df=train_df,
            test_df=test_df,
            train_dir=train_dir,
            test_dir=test_dir,
            output_dir=output_dir,
            device=device,
            args=args,
        )
        summaries.append(summary)
        experiment_weights.append(summary["oof_accuracy"])

    weights = np.array(experiment_weights, dtype=np.float64)
    weights = weights / weights.sum()

    ensemble_oof = np.zeros((len(train_df), NUM_CLASSES), dtype=np.float32)
    ensemble_test = np.zeros((len(test_df), NUM_CLASSES), dtype=np.float32)
    for weight, summary in zip(weights, summaries):
        exp_dir = output_dir / summary["experiment_name"]
        ensemble_oof += weight * np.load(exp_dir / "oof_probs.npy")
        ensemble_test += weight * np.load(exp_dir / "test_probs.npy")

    ensemble_preds = ensemble_oof.argmax(axis=1)
    ensemble_acc = accuracy_score(train_df["target"].to_numpy(), ensemble_preds)
    ensemble_cm = confusion_matrix(train_df["target"].to_numpy(), ensemble_preds)
    plot_confusion(
        ensemble_cm,
        labels=["0 person", "1 person", "2 persons", "3 persons"],
        path=output_dir / "ensemble_confusion_matrix.png",
        title="Weighted ensemble OOF confusion matrix",
    )

    np.save(output_dir / "ensemble_oof_probs.npy", ensemble_oof)
    np.save(output_dir / "ensemble_test_probs.npy", ensemble_test)

    submission = test_df.copy()
    submission["target"] = ensemble_test.argmax(axis=1).astype(int)
    submission_path = output_dir / "submission_ensemble.csv"
    submission.to_csv(submission_path, index=False)

    save_json(
        {
            "device": str(device),
            "seed": args.seed,
            "experiments": summaries,
            "ensemble_weights": [
                {"experiment_name": summary["experiment_name"], "weight": float(weight)}
                for summary, weight in zip(summaries, weights)
            ],
            "ensemble_oof_accuracy": float(ensemble_acc),
            "submission_path": str(submission_path),
        },
        output_dir / "run_summary.json",
    )

    print(json.dumps({"ensemble_oof_accuracy": ensemble_acc, "submission_path": str(submission_path)}, indent=2))


if __name__ == "__main__":
    main()
