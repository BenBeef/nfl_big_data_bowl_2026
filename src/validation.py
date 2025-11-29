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


def compute_val_rmse_multi_player_stt(model, X_val_multi, y_val_multi_dx, y_val_multi_dy, player_masks, horizon, device):
    """
    多球员模型验证 - 计算被追踪球员的 RMSE
    
    使用player_masks来mask掉：
    1. 非被追踪球员（pad）
    2. 没有真实数据的horizon位置
    
    Shape分析:
    ───────────────────────────────────────────────────
    输入X_val_multi: List[N] of (22, seq_len, input_dim)
    输入y_val_multi_dx: List[N] of (22, horizon)
    输入y_val_multi_dy: List[N] of (22, horizon)
    输入player_masks: List[N] of (22, horizon)  ← 有效位置mask
    ───────────────────────────────────────────────────
    模型输出:
    predict: (N, 22, horizon, 2)
    ───────────────────────────────────────────────────
    Mask包含:
    - player i 在 horizon j 处有有效数据 → mask[i,j] = 1.0
    - player i 是pad或超出数据长度 → mask[i,j] = 0.0
    ───────────────────────────────────────────────────
    RMSE计算:
    - 只在 mask==1.0 的位置计算误差
    - rmse = sqrt(sum(se * mask) / (2 * sum(mask)))
    ───────────────────────────────────────────────────
    """
    model.eval()
    
    if isinstance(X_val_multi, list):
        X_val_multi = np.stack(X_val_multi).astype(np.float32)
    
    X_t = torch.tensor(X_val_multi, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # 预测: (batch, 22, horizon, 2)
        predict = model(X_t).cpu().numpy()
    
    # 准备目标和mask
    batch_targets = []
    batch_masks = []
    
    for players_dx, players_dy, mask in zip(y_val_multi_dx, y_val_multi_dy, player_masks):
        dx_t = torch.tensor(players_dx, dtype=torch.float32)
        dy_t = torch.tensor(players_dy, dtype=torch.float32)
        player_target = torch.stack([dx_t, dy_t], dim=-1)
        batch_targets.append(player_target.numpy())
        batch_masks.append(mask)  # ← 使用传入的mask（记录有效数据位置）
    
    target = np.stack(batch_targets)  # (batch, 22, horizon, 2)
    combined_mask = np.stack(batch_masks)  # (batch, 22, horizon) - 记录有效数据位置
    
    # 计算 RMSE (只计算有效位置)
    pdx = predict[..., 0]  # (batch, 22, horizon)
    pdy = predict[..., 1]
    ydx = target[..., 0]
    ydy = target[..., 1]
    
    # 2D 误差，用mask加权（只在有有效数据的位置计算）
    se_sum2d = ((pdx - ydx) ** 2 + (pdy - ydy) ** 2) * combined_mask
    denom = combined_mask.sum() + 1e-8
    
    rmse = np.sqrt(se_sum2d.sum() / (2.0 * denom))
    
    return float(rmse)
