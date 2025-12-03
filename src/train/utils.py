import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.segmentation import DiceScore
import sys

from src.models.early_stopping import EarlyStopping
from src.models.metrics import dice


def train(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, es: EarlyStopping, device: torch.device, num_epochs: int, num_classes: int = 3) -> tuple[torch.nn.Module, int]:
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        for x_batch, a_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            a_batch = a_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch, a_batch)
            loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)
            train_dice += dice(y_pred, y_batch, num_classes) * x_batch.size(0)
        train_loss = train_loss / len(train_loader.dataset)
        train_dice = train_dice / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for x_batch, a_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                a_batch = a_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(x_batch, a_batch)
                val_loss += criterion(y_pred, y_batch).item() * x_batch.size(0)
                val_dice += dice(y_pred, y_batch, num_classes) * x_batch.size(0)
        val_loss = val_loss / len(val_loader.dataset)
        val_dice = val_dice / len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f} - Val Loss: {val_loss:.4f} - Val Dice: {val_dice:.4f}")

        es(val_loss, model)
        if es.early_stop:
            print("⏹️ Early stopping")
            break
    
    epoch = es.epoch - es.epochs_without_improvement
    return model, epoch

def eval(model: torch.nn.Module, train_loader: DataLoader, test_loader: DataLoader, val_loader: DataLoader, criterion: torch.nn.Module, device: torch.device, num_classes: int = 3) -> dict[str, float]:
    metrics = {}
    model.eval()
    with torch.no_grad():
        for set, loader in [("train", train_loader), ("test", test_loader), ("val", val_loader)]:
            total_loss = 0.
            total_dice = 0.
            for x_batch, a_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                a_batch = a_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(x_batch, a_batch)
                
                loss = criterion(y_pred, y_batch)
                total_loss += loss.item() * x_batch.size(0)
                total_dice += dice(y_pred, y_batch, num_classes).item() * x_batch.size(0)

            metrics[f"{set}_loss"] = total_loss / len(loader.dataset)
            metrics[f"{set}_dice"] = total_dice / len(loader.dataset)
    return metrics
