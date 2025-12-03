import torch
import numpy as np
from torch.distributions.kl import _x_log_x
from src.config import Config


# TODO: TTA?
def predict_sst(model, scaler, X_test_raw, device):
    model.eval()
    outs_dx, outs_dy = [], []

    base = np.stack([scaler.transform(s) for s in X_test_raw]).astype(np.float32)
    xt = torch.tensor(base, device=device)

    with torch.no_grad():
        output = model(xt)

        dx = output[:, :, 0]  # 第一维为 dx
        dy = output[:, :, 1]  # 第二维为 dy

    outs_dx.append(dx.detach().cpu().numpy())
    outs_dy.append(dy.detach().cpu().numpy())

    return np.mean(outs_dx, axis=0), np.mean(outs_dy, axis=0)


def predict_multi_player(model, scaler, X_test_raw, device, seq_meta_multi=None, play_rel_seqs=None, rel_scaler=None, play_masks=None):
    """
    多球员模型推理（支持单球员和多球员输入）
    
    Args:
        model: MultiPlayerGRUTransformer 模型
        scaler: StandardScaler
        X_test_raw: List[(seq_len, n_features)] - 单球员序列，会自动转为多球员格式
        device: 设备
        play_rel_seqs: 相对特征（暂不使用）
        rel_scaler: 相对特征标准化器（暂不使用）
        play_masks: 追踪mask（暂不使用）
    
    Returns:
        (outs_dx, outs_dy) - 预测结果 (batch, horizon)
    """
    model.eval()
    
    # 将单球员序列堆叠后，包装成多球员格式（batch=1）
    base = np.stack([scaler.transform(s.reshape(-1, s.shape[-1])).reshape(s.shape) for s in X_test_raw]).astype(np.float32)  # (n_samples, seq_len, n_feat)
    xt = torch.tensor(base, device=device)

    play_rel_seqs = [play_rel_seqs[i][:, -1:, :] for i in range(len(play_rel_seqs))]  if play_rel_seqs is not None else None # (22, 1, 2*n_players) - 只要最后一帧
    rel = np.stack([rel_scaler.transform(s.reshape(-1, s.shape[-1])).reshape(s.shape) for s in play_rel_seqs]).astype(np.float32)  # (n_samples, seq_len, n_feat)
    bx_rel = torch.tensor(rel, device=device)


    btm_list = (play_masks.sum(axis=-1) > 0).astype(np.float32)
    btm = torch.tensor(np.stack(btm_list), dtype=torch.float32)

    xt, btm, bx_rel = xt.to(Config.DEVICE), btm.to(Config.DEVICE), bx_rel.to(Config.DEVICE)
    
    with torch.no_grad():
        output = model(xt, tracked_mask=btm, x_relative=bx_rel)
        # output = model(xt)  # STTransformer: (n_samples, horizon, 2)
        
        dx = output[:, :, :, 0]  # (n_samples, horizon)
        dy = output[:, :, :, 1]
    

    b, n, h = dx.shape
    dx = dx.reshape(b * n, h)
    dy = dy.reshape(b * n, h)
    
    outs_dx = dx.detach().cpu().numpy().tolist()
    outs_dy = dy.detach().cpu().numpy().tolist()
    
    valid_idx = []
    origin_metas_list = []
    for meta in seq_meta_multi:
        max_players = meta['max_players']
        valid_players = meta['n_players']
        origin_metas = meta['origin_metas']
        origin_metas_list.extend(origin_metas)
        for i in range(max_players):
            if i < valid_players:
                valid_idx.append(1)
            else:
                valid_idx.append(0)

    outs_dx = np.array([v for i, v in enumerate(outs_dx) if valid_idx[i]], dtype=np.float32)
    outs_dy = np.array([v for i, v in enumerate(outs_dy) if valid_idx[i]], dtype=np.float32)

    return outs_dx, outs_dy, origin_metas_list