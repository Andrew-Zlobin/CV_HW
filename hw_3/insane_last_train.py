import torch
import numpy as np

@torch.no_grad()
def segmentation_metrics(logits, targets, threshold=0.5, eps=1e-7):
    
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) - intersection

    iou = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (
        preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + eps
    )

    pixel_acc = (preds == targets).float().mean(dim=(1,2,3))

    return {
        "iou": iou.mean().item(),
        "dice": dice.mean().item(),
        "acc": pixel_acc.mean().item()
    }


def train_one_epoch(
    model, loader, optimizer, criterion, device
):
    model.train()

    total_loss = 0.0
    metrics_sum = {"iou": 0.0, "dice": 0.0, "acc": 0.0}

    for x, y in loader:
        x, y = x.to(device), y.unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        logits = model(x)

        # print(logits.shape, y.shape)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        batch_metrics = segmentation_metrics(logits, y)
        for k in metrics_sum:
            metrics_sum[k] += batch_metrics[k] * x.size(0)

    n = len(loader.dataset)
    return (
        total_loss / n,
        {k: v / n for k, v in metrics_sum.items()}
    )



@torch.no_grad()
def validate(
    model, loader, criterion, device
):
    model.eval()

    total_loss = 0.0
    metrics_sum = {"iou": 0.0, "dice": 0.0, "acc": 0.0}

    for x, y in loader:
        x, y = x.to(device), y.unsqueeze(1).to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)

        batch_metrics = segmentation_metrics(logits, y)
        for k in metrics_sum:
            metrics_sum[k] += batch_metrics[k] * x.size(0)

    n = len(loader.dataset)
    return (
        total_loss / n,
        {k: v / n for k, v in metrics_sum.items()}
    )

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs
):
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_iou": [],
        "val_iou": [],
        "train_dice": [],
        "val_dice": [],
        "train_acc": [],
        "val_acc": [],
        # "train_metrics": [],
        # "val_metrics": []
        
    }

    for epoch in range(epochs):
        tr_loss, tr_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device
        )
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        for metric in tr_metrics:
            history["train_" + metric].append(tr_metrics[metric])
        for metric in val_metrics:
            history["val_" + metric].append(val_metrics[metric])
        # history["train_metrics"].append(tr_metrics)
        # history["val_metrics"].append(val_metrics)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train IoU: {tr_metrics['iou']:.3f}, "
            f"Val IoU: {val_metrics['iou']:.3f}, "
            f"Val Dice: {val_metrics['dice']:.3f}, "
            f"Val Pixel Acc: {val_metrics['acc']:.3f}"
        )

    return history

