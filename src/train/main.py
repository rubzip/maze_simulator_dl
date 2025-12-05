import os
import json

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.unet import GameUNet
from src.models.early_stopping import EarlyStopping
from .utils import train, eval
from settings import WIDTH, WALL_PROB

BATCH_SIZE = 32
NUM_EPOCHS = 1_000
NUM_CLASSES = 3

L2 = 1e-5
LR = 1e-3
PATIENCE = 10

TRAIN = "./dataset/train.pt"
TEST = "./dataset/test.pt"
VAL = "./dataset/val.pt"

MODEL_FOLDER = "./model/"

SURFACE = WIDTH * WIDTH
CLASS_FREQ = np.array([(1 - WALL_PROB) * SURFACE - 1, WALL_PROB * SURFACE, 1])
class_weight = CLASS_FREQ.sum() / CLASS_FREQ

if __name__ == "__main__":
    train_ds = torch.load(TRAIN, weights_only=False)
    test_ds = torch.load(TEST, weights_only=False)
    val_ds = torch.load(VAL, weights_only=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GameUNet(n_classes=3, n_actions=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2)
    weights_tensor = torch.tensor(class_weight, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor).to(device)
    es = EarlyStopping(patience=PATIENCE)

    model, epoch = train(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, criterion=criterion, es=es, device=device, num_epochs=NUM_EPOCHS, num_classes=NUM_CLASSES)
    metrics = eval(model=model, train_loader=train_loader, test_loader=test_loader, val_loader=val_loader, criterion=criterion, device=device, num_classes=NUM_CLASSES)

    os.makedirs(MODEL_FOLDER, exist_ok=True)
    with open(MODEL_FOLDER + "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    torch.save(model.state_dict(), MODEL_FOLDER + 'model.pth')
