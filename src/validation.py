import torch
import numpy as np

from .utils import prepare_targets_stt


def diagnose_rmse_issue(predict, target, combined_mask):
    """
    诊断 RMSE 计算中的问题
    """
    print("\n[RMSE诊断]")
    print(f"  预测形状: {predict.shape}")
    print(f"  目标形状: {target.shape}")
    print(f"  Mask 形状: {combined_mask.shape}")
    
    # 检查预测值的分布
    pdx = predict[..., 0]
    pdy = predict[..., 1]
    print(f"  预测值范围: [{pdx.min():.4f}, {pdx.max():.4f}]")
    print(f"  目标值范围: [{target[..., 0].min():.4f}, {target[..., 0].max():.4f}]")
    
    # 检查填充球员的预测
    print(f"\n  Mask 统计:")
    print(f"    有效位置数: {combined_mask.sum():.0f}")
    print(f"    总位置数: {combined_mask.size}")
    print(f"    有效比率: {combined_mask.sum() / combined_mask.size * 100:.2f}%")
    
    # 计算误差
    se_2d = (pdx - target[..., 0]) ** 2 + (pdy - target[..., 1]) ** 2
    se_masked = se_2d * combined_mask
    print(f"\n  误差统计:")
    print(f"    SE 范围: [{se_2d.min():.6f}, {se_2d.max():.6f}]")
    print(f"    Masked SE 总和: {se_masked.sum():.6f}")
    
    # 计算不同的 RMSE
    rmse_standard = np.sqrt(se_masked.sum() / (2.0 * combined_mask.sum() + 1e-8))
    print(f"\n  标准 RMSE: {rmse_standard:.6f}")
    
    # 如果没有 mask，会是什么？
    se_no_mask = se_2d.sum()
    rmse_no_mask = np.sqrt(se_no_mask / (2.0 * se_2d.size + 1e-8))
    print(f"  无 Mask RMSE: {rmse_no_mask:.6f} (包含填充球员)")
    print("  " + "="*60)


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
    
    ⚠️ 关键：player_masks 在这里用于过滤预测值，而不仅仅是目标值
    """
    model.eval()
    
    if isinstance(X_val_multi, list):
        X_val_multi = np.stack(X_val_multi).astype(np.float32)
    
    X_t = torch.tensor(X_val_multi, dtype=torch.float32).to(device)
    
    # 从 player_masks 推导 tracked_mask（每个球员是否被追踪）
    # 如果某个球员的任何帧被标记为 1，就说明该球员被追踪
    batch_targets = []
    batch_masks = []
    batch_tracked_masks = []
    
    for players_dx, players_dy, mask in zip(y_val_multi_dx, y_val_multi_dy, player_masks):
        dx_t = torch.tensor(players_dx, dtype=torch.float32)
        dy_t = torch.tensor(players_dy, dtype=torch.float32)
        player_target = torch.stack([dx_t, dy_t], dim=-1)
        batch_targets.append(player_target.numpy())
        batch_masks.append(mask)  # ← 记录有效数据位置 (22, horizon)
        
        # 推导 tracked_mask: 检查每个球员是否有任何有效帧 (1 表示被追踪)
        tracked = (mask.sum(axis=1) > 0).astype(np.float32)  # (22,)
        batch_tracked_masks.append(tracked)
    
    target = np.stack(batch_targets)  # (batch, 22, horizon, 2)
    combined_mask = np.stack(batch_masks)  # (batch, 22, horizon) - 记录有效数据位置
    tracked_mask = np.stack(batch_tracked_masks)  # (batch, 22) - 记录被追踪球员
    
    # 将 tracked_mask 转换为 torch tensor 并传入模型
    tracked_mask_t = torch.tensor(tracked_mask, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # ⚠️ 关键：在验证时也要传入 tracked_mask，保持与训练一致
        predict = model(X_t, tracked_mask=tracked_mask_t).cpu().numpy()
    
    # 计算 RMSE (只计算有效位置)
    pdx = predict[..., 0]  # (batch, 22, horizon)
    pdy = predict[..., 1]
    ydx = target[..., 0]
    ydy = target[..., 1]
    
    # 2D 误差，用mask加权（只在有有效数据的位置计算）
    # ⚠️ 关键：这里 mask 应用于预测误差，过滤掉填充球员的垃圾预测
    se_sum2d = ((pdx - ydx) ** 2 + (pdy - ydy) ** 2) * combined_mask
    denom = combined_mask.sum() + 1e-8
    
    rmse = np.sqrt(se_sum2d.sum() / (2.0 * denom))
    
    # 诊断输出
    diagnose_rmse_issue(predict, target, combined_mask)
    
    return float(rmse)
