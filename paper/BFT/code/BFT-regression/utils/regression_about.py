import torch
import numpy as np
from sklearn.metrics import r2_score


def regression_metrics(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # r2 = r2_score(y_true, y_pred)
    cc = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    return cc, rmse, mae

