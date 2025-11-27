import torch
import numpy as np

from .utils import prepare_targets_stt


def compute_val_rmse_stt(model, X_val_sc, ydx_list, ydy_list, horizon, device):
    # stack list -> np.array
    if isinstance(X_val_sc, list):
        X_val_sc = np.stack(X_val_sc).astype(np.float32)

    X_t = torch.tensor(X_val_sc, dtype=torch.float32).to(device)

    with torch.no_grad():
        predict = model(X_t).cpu().numpy()  # [B, H, 2]

    # targets & mask
    by, bm = prepare_targets_stt(ydx_list, ydy_list, horizon)
    if torch.is_tensor(by):
        by = by.numpy()
    if torch.is_tensor(bm):
        bm = bm.numpy()

    pdx, pdy = predict[..., 0], predict[..., 1]
    ydx, ydy = by[..., 0], by[..., 1]
    mask = bm

    # squared error
    se_sum2d = ((pdx - ydx) ** 2 + (pdy - ydy) ** 2) * mask
    denom = mask.sum() + 1e-8

    return float(np.sqrt(se_sum2d.sum() / (2.0 * denom)))


def _prepare_targets_1d(batch_1d, max_h):
    """为单一维度(x或y)准备目标，返回shape (B, H)"""
    tensors, masks = [], []
    for vals in batch_1d:
        L = len(vals)
        padded = np.pad(vals, (0, max_h - L), constant_values=0).astype(np.float32)
        mask = np.zeros(max_h, dtype=np.float32)
        mask[:L] = 1.0
        tensors.append(torch.tensor(padded))
        masks.append(torch.tensor(mask))
    
    return torch.stack(tensors), torch.stack(masks)  # (B, H), (B, H)


def compute_val_rmse_separate_models_stt(model_x, model_y, X_val_sc, ydx_list, ydy_list, horizon, device):
    """
    使用分离的 X 和 Y 模型进行验证，计算组合的 RMSE
    """
    # stack list -> np.array
    if isinstance(X_val_sc, list):
        X_val_sc = np.stack(X_val_sc).astype(np.float32)

    X_t = torch.tensor(X_val_sc, dtype=torch.float32).to(device)

    with torch.no_grad():
        # 分别从两个模型获取预测
        predict_x = model_x(X_t).cpu().numpy()  # [B, H]
        predict_y = model_y(X_t).cpu().numpy()  # [B, H]

    # 获取目标和 mask
    by_x, bm_x = _prepare_targets_1d(ydx_list, horizon)
    by_y, bm_y = _prepare_targets_1d(ydy_list, horizon)
    
    if torch.is_tensor(by_x):
        by_x = by_x.numpy()
    if torch.is_tensor(by_y):
        by_y = by_y.numpy()
    if torch.is_tensor(bm_x):
        bm_x = bm_x.numpy()
    if torch.is_tensor(bm_y):
        bm_y = bm_y.numpy()

    ydx = by_x  # [B, H]
    ydy = by_y  # [B, H]
    pdx = predict_x  # [B, H]
    pdy = predict_y  # [B, H]
    
    # 使用相同的 mask（两个维度应该一致）
    mask = bm_x

    # 计算 2D 欧几里得距离的平方误差
    se_sum2d = ((pdx - ydx) ** 2 + (pdy - ydy) ** 2) * mask
    denom = mask.sum() + 1e-8

    return float(np.sqrt(se_sum2d.sum() / (2.0 * denom)))
