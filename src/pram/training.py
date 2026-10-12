"""BCE training with validation-loss early stopping, as in the main cells."""

import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import RawRejectMemory


def set_seed(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested="auto"):
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


@torch.no_grad()
def predict_batches(model, prototypes, values, batch_size=10000):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.eval()
    device = next(model.parameters()).device
    prototypes = prototypes.to(device)
    result = []
    for start in range(0, len(values), batch_size):
        batch = torch.as_tensor(values[start:start + batch_size], dtype=torch.float32, device=device)
        logits, _, _ = model(batch, prototypes)
        result.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(result) if result else np.empty(0, dtype=np.float32)


def train_model(data, prototypes, config, device):
    def loader(X, y, shuffle):
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle)

    train_loader = loader(data.X_train, data.y_train, True)
    val_loader = loader(data.X_val, data.y_val, False)
    model = RawRejectMemory().to(device)
    memory = torch.tensor(prototypes, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    best_loss, best_state, best_epoch, bad_epochs = np.inf, None, 0, 0
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss, n = 0.0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _, _ = model(X, memory)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
            n += len(y)
        model.eval()
        losses = []
        with torch.no_grad():
            for X, y in val_loader:
                logits, _, _ = model(X.to(device), memory)
                losses.append(criterion(logits, y.to(device)).item())
        # Preserve the notebook's unweighted mean of per-batch validation means.
        val_loss = float(np.mean(losses))
        if not np.isfinite(val_loss):
            raise ValueError("Non-finite validation loss; check features and learning rate")
        history.append({"epoch": epoch, "train_loss": train_loss / n, "validation_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad_epochs += 1
        if bad_epochs >= config.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history, {"best_epoch": best_epoch, "best_validation_loss": best_loss,
                            "epochs_completed": len(history)}

