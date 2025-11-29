import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from transformers import get_cosine_schedule_with_warmup

from .config import Config
from .utils import (
    prepare_targets_stt,
    save_fold_artifacts_stt,
)
from .validation import compute_val_rmse_stt, compute_val_rmse_multi_player_stt


class TemporalHuber(nn.Module):
    def __init__(self, delta=0.5, time_decay=0.02, lam_smooth=0.01):
        super().__init__()
        self.delta = delta
        self.time_decay = time_decay
        self.lam_smooth = lam_smooth

    def forward(self, pred, target, mask):
        # base huber
        err = pred - target
        abs_err = torch.abs(err)
        huber = torch.where(
            abs_err <= self.delta,
            0.5 * err * err,
            self.delta * (abs_err - 0.5 * self.delta),
        )

        # time decay
        if self.time_decay and self.time_decay > 0:
            L = pred.size(1)
            t = torch.arange(L, device=pred.device, dtype=pred.dtype)
            w = torch.exp(-self.time_decay * t).view(1, L, 1)
            huber = huber * w
            mask = mask.unsqueeze(-1) * w

        main_loss = (huber * mask).sum() / (mask.sum() + 1e-8)

        # # velocity smooth
        # if self.lam_smooth and pred.size(1) > 2:
        #     d1 = pred[:, 1:] - pred[:, :-1]
        #     d2 = d1[:, 1:] - d1[:, :-1]
        #     m2 = mask[:, 2:]
        #     smooth = (d2 * d2) * m2
        #     smooth_loss = smooth.sum() / (m2.sum() + 1e-8)
        # else:
        #     smooth_loss = pred.new_tensor(0.0)

        return main_loss


def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_pos = max_position

    def forward(self, x):
        # x: [B, T, H]
        t = torch.arange(x.size(1), device=x.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        sin, cos = freqs.sin(), freqs.cos()
        sin = torch.repeat_interleave(sin, 2, dim=1)
        cos = torch.repeat_interleave(cos, 2, dim=1)
        return x * cos.unsqueeze(0) + rotate_half(x) * sin.unsqueeze(0)


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.net(x) + x)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super().__init__()
        layers = []

        # 输入层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))

        # 隐藏层（带残差连接）
        for _ in range(num_layers - 2):
            layers.append(ResidualBlock(hidden_dim, hidden_dim, dropout))

        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TemporalTransformerEncoder(nn.Module):
    """
    时序 Transformer 编码器
    用于处理单个球员的序列，在帧维度（horizon）上做 attention
    
    输入: (batch, seq_len, input_dim)
    输出: (batch, hidden_dim)
    """
    
    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = Config.HIDDEN_DIM
        self.n_heads = Config.N_HEADS
        self.n_layers = Config.N_LAYERS
        self.n_querys = Config.N_QUERYS
        
        # 1. 特征投影
        self.input_projection = nn.Linear(input_dim, self.hidden_dim)
        
        # 2. 位置编码
        self.pos_embed = nn.Parameter(
            torch.randn(1, Config.WINDOW_SIZE, self.hidden_dim)
        )
        self.embed_dropout = nn.Dropout(dropout)
        
        # 3. Transformer Encoder（在时序维度）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers
        )
        
        # 4. Attention Pooling
        self.pool_ln = nn.LayerNorm(self.hidden_dim)
        self.pool_attn = nn.MultiheadAttention(
            self.hidden_dim, num_heads=self.n_heads, batch_first=True
        )
        self.pool_query = nn.Parameter(torch.randn(1, self.n_querys, self.hidden_dim))
        
        # 5. 汇聚投影层
        self.pool_proj = nn.Linear(self.n_querys * self.hidden_dim, self.hidden_dim)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, seq_len, input_dim)
        
        Returns:
            output: (batch, hidden_dim) - 时序汇聚后的特征
        """
        B, T, _ = x.shape
        
        # 投影到 hidden_dim
        x_embed = self.input_projection(x)  # (B, T, hidden_dim)
        
        # 加上位置编码
        x = x_embed + self.pos_embed[:, :T, :]
        x = self.embed_dropout(x)
        
        # Transformer 编码
        h = self.transformer_encoder(x)  # (B, T, hidden_dim)
        
        # Attention pooling：用可学习的 query 从时序序列中汇聚信息
        q = self.pool_query.expand(B, -1, -1)  # (B, n_querys, hidden_dim)
        ctx, _ = self.pool_attn(q, self.pool_ln(h), self.pool_ln(h))  # (B, n_querys, hidden_dim)
        output = ctx.flatten(start_dim=1)  # (B, n_querys * hidden_dim)
        
        # 投影回 hidden_dim
        # output = F.gelu(self.pool_proj(ctx))  # (B, hidden_dim)
        
        return output  # (B, hidden_dim)


class STTransformer(nn.Module):
    """
    Spatio-Temporal Transformer
    """

    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = Config.MAX_FUTURE_HORIZON
        self.hidden_dim = Config.HIDDEN_DIM
        self.n_heads = Config.N_HEADS
        self.n_layers = Config.N_LAYERS
        self.n_querys = Config.N_QUERYS

        # 1. Spatio: 特征嵌入
        self.input_projection = nn.Linear(input_dim, self.hidden_dim)

        # 2. Temporal: 可学习的位置编码
        self.pos_embed = nn.Parameter(
            torch.randn(1, Config.WINDOW_SIZE, self.hidden_dim)
        )
        # self.rope = RotaryEmbedding(self.hidden_dim, Config.WINDOW_SIZE)
        self.embed_dropout = nn.Dropout(dropout)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers
        )

        # 4. Pooling
        self.pool_ln = nn.LayerNorm(self.hidden_dim)
        self.pool_attn = nn.MultiheadAttention(
            self.hidden_dim, num_heads=self.n_heads, batch_first=True
        )
        self.pool_query = nn.Parameter(torch.randn(1, self.n_querys, self.hidden_dim))

        # 5. 输出 Head
        self.head = ResidualMLP(
            input_dim=self.n_querys * self.hidden_dim,
            hidden_dim=Config.MLP_HIDDEN_DIM,
            output_dim=self.horizon * 2,
            num_layers=Config.N_RES_BLOCKS,
            dropout=0.2,
        )

    def forward(self, x: torch.Tensor):
        # [batch, temporal, spatio]
        # TODO: another approprate forward [batch, player, temporal, spatio] ==> [batch, player, temporal, 2]?
        B, T, _ = x.shape

        x_embed = self.input_projection(x)
        # x = self.rope(x_embed)
        x = x_embed + self.pos_embed[:, :T, :]
        x = self.embed_dropout(x)

        h = self.transformer_encoder(x)

        q = self.pool_query.expand(B, -1, -1)
        ctx, _ = self.pool_attn(q, self.pool_ln(h), self.pool_ln(h))
        ctx = ctx.flatten(start_dim=1)

        out = self.head(ctx)
        out = out.view(B, self.horizon, 2)

        out = torch.cumsum(out, dim=1)

        return out


def _build_train_batches(X_train, y_train_dx, y_train_dy, shuffle=True):
    """构建训练批次（支持 shuffle）"""
    indices = np.arange(len(X_train))
    if shuffle:
        np.random.shuffle(indices)
    
    train_batches = []
    for i in range(0, len(X_train), Config.BATCH_SIZE):
        end = min(i + Config.BATCH_SIZE, len(X_train))
        batch_indices = indices[i:end]
        
        X_batch = [X_train[j] for j in batch_indices]
        y_dx_batch = [y_train_dx[j] for j in batch_indices]
        y_dy_batch = [y_train_dy[j] for j in batch_indices]
        
        bx = torch.tensor(np.stack(X_batch).astype(np.float32))
        by, bm = prepare_targets_stt(
            y_dx_batch,
            y_dy_batch,
            Config.MAX_FUTURE_HORIZON,
        )
        train_batches.append((bx, by, bm))
    
    return train_batches


def _build_val_batches(X_val, y_val_dx, y_val_dy):
    """构建验证批次（不需要 shuffle）"""
    val_batches = []
    for i in range(0, len(X_val), Config.BATCH_SIZE):
        end = min(i + Config.BATCH_SIZE, len(X_val))
        bx = torch.tensor(np.stack(X_val[i:end]).astype(np.float32))
        by, bm = prepare_targets_stt(
            [y_val_dx[j] for j in range(i, end)],
            [y_val_dy[j] for j in range(i, end)],
            Config.MAX_FUTURE_HORIZON,
        )
        val_batches.append((bx, by, bm))
    
    return val_batches


def train_model_stt(
    X_train,
    y_train_dx,
    y_train_dy,
    X_val,
    y_val_dx,
    y_val_dy,
    input_dim,
):
    device = Config.DEVICE

    # Construct val_batches（只构建一次，不需要 shuffle）
    val_batches = _build_val_batches(X_val, y_val_dx, y_val_dy)

    # Define model, criterion, optimizer, scheduler
    model = STTransformer(
        input_dim=input_dim,
    ).to(device)
    criterion = TemporalHuber(delta=0.5, time_decay=0.03)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5
    )
    # total_steps = Config.EPOCHS * len(train_batches)
    # warmup_steps = int(0.1 * total_steps)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    best_loss, best_state, bad = float("inf"), None, 0
    start_time = time.time()

    for epoch in range(1, Config.EPOCHS + 1):
        # 每个 epoch 都重新 shuffle 并构建训练批次
        train_batches = _build_train_batches(X_train, y_train_dx, y_train_dy, shuffle=True)
        
        model.train()
        train_losses = []
        for bx, by, bm in train_batches:
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)
            pred = model(bx)
            loss = criterion(pred, by, bm)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            # scheduler.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by, bm in val_batches:
                bx, by, bm = bx.to(device), by.to(device), bm.to(device)
                pred = model(bx)
                val_losses.append(criterion(pred, by, bm).item())

        train_loss, val_loss = np.mean(train_losses), np.mean(val_losses)
        scheduler.step(val_loss)

        if epoch % 10 == 0:
            total_time = time.time() - start_time
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)
            print(
                f"  Epoch {epoch:>3}: train={train_loss:.4f}, val={val_loss:.4f}, "
                f"Time_elapsed={minutes:>2}min {seconds:>2}s"
            )

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= Config.PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)

    return model, best_loss


def train_all_folds_stt(
    gkf, sequences, groups, targets_dx, targets_dy, seed, input_dim
):
    """
    旧版本：单球员训练（已弃用）
    """
    print("[WARN] 使用旧版本 train_all_folds_stt（单球员）")
    fold_rmses = []
    all_rmse = []
    cv_log = []

    for fold, (tr, va) in enumerate(gkf.split(sequences, y=None, groups=groups), 1):
        print(f"\n{'-'*60}\nFold {fold}/{Config.N_FOLDS} (seed {seed})\n{'-'*60}")

        X_tr = [sequences[i] for i in tr]
        X_va = [sequences[i] for i in va]
        y_tr_dx = [targets_dx[i] for i in tr]
        y_va_dx = [targets_dx[i] for i in va]
        y_tr_dy = [targets_dy[i] for i in tr]
        y_va_dy = [targets_dy[i] for i in va]

        scaler = StandardScaler()
        scaler.fit(np.vstack([s for s in X_tr]))

        X_tr_sc = [scaler.transform(s) for s in X_tr]
        X_va_sc = [scaler.transform(s) for s in X_va]

        model, loss = train_model_stt(
            X_tr_sc,
            y_tr_dx,
            y_tr_dy,
            X_va_sc,
            y_va_dx,
            y_va_dy,
            input_dim,
        )

        rmse = compute_val_rmse_stt(
            model,
            X_va_sc,
            [targets_dx[i] for i in va],
            [targets_dy[i] for i in va],
            Config.MAX_FUTURE_HORIZON,
            Config.DEVICE,
        )

        print(
            f"[VAL] seed {seed} fold {fold} → "
            f"Huber loss={loss:.5f} | "
            f"RMSE={rmse:.4f}"
        )

        fold_rmses.append(rmse)
        all_rmse.append(rmse)
        cv_log.append(
            {
                "seed": seed,
                "fold": fold,
                "rmse": rmse,
                "loss": float(loss),
            }
        )

        # Save model
        save_fold_artifacts_stt(
            seed=seed,
            fold=fold,
            scaler=scaler,
            model=model,
            base_dir=Config.SAVE_DIR,
        )

    print(
        f"[SEED SUMMARY] seed {seed} RMSEs: {[f'{r:.4f}' for r in fold_rmses]} | "
        f"mean={float(np.mean(fold_rmses)):.4f} yards"
    )

    return all_rmse, cv_log


def train_all_folds_multi_player_stt(
    gkf, sequences, groups, targets_dx, targets_dy, player_mask, seed, input_dim
):
    """
    新版本：多球员模型训练
    
    Args:
        gkf: GroupKFold分割器
        sequences: List[(22, seq_len, n_features)] - 多球员格式
        groups: 分组标签 (game_id)
        targets_dx: List[(22, horizon)] - 多球员位移
        targets_dy: List[(22, horizon)]
        player_mask: List[(22, horizon)] - 每个player的有效位置mask
        seed: 随机种子
        input_dim: 特征维度 (167)
    
    Returns:
        all_rmse: 所有fold的RMSE列表
        cv_log: 交叉验证日志
    """
    fold_rmses = []
    all_rmse = []
    cv_log = []
    
    print(f"\n{'='*70}")
    print(f"多球员模型训练 - MultiPlayerGRUTransformer")
    print(f"{'='*70}")

    for fold, (tr, va) in enumerate(gkf.split(sequences, y=None, groups=groups), 1):
        print(f"\n{'-'*70}\nFold {fold}/{Config.N_FOLDS} (seed {seed})\n{'-'*70}")

        # 多球员数据格式
        X_tr = [sequences[i] for i in tr]  # List[(22, seq_len, n_features)]
        X_va = [sequences[i] for i in va]
        y_tr_dx = [targets_dx[i] for i in tr]  # List[(22, horizon)]
        y_va_dx = [targets_dx[i] for i in va]
        y_tr_dy = [targets_dy[i] for i in tr]
        y_va_dy = [targets_dy[i] for i in va]
        mask_tr = [player_mask[i] for i in tr]  # ← 训练集的mask
        mask_va = [player_mask[i] for i in va]  # ← 验证集的mask

        # Scaler: 需要处理多球员数据
        # 将所有序列和球员展平用于fit
        X_tr_flat = []
        for seq in X_tr:  # seq: (22, seq_len, n_features)
            for player_seq in seq:  # player_seq: (seq_len, n_features)
                X_tr_flat.append(player_seq)
        
        scaler = StandardScaler()
        scaler.fit(np.vstack(X_tr_flat))

        # 对每个序列的每个球员应用scaler
        X_tr_sc = []
        for seq in X_tr:  # seq: (22, seq_len, n_features)
            seq_scaled = np.zeros_like(seq, dtype=np.float32)
            for player_idx in range(Config.MAX_NUM_PLAYER):
                seq_scaled[player_idx] = scaler.transform(seq[player_idx])
            X_tr_sc.append(seq_scaled)
        
        X_va_sc = []
        for seq in X_va:
            seq_scaled = np.zeros_like(seq, dtype=np.float32)
            for player_idx in range(Config.MAX_NUM_PLAYER):
                seq_scaled[player_idx] = scaler.transform(seq[player_idx])
            X_va_sc.append(seq_scaled)

        # 训练多球员模型
        print(f"  Training {len(X_tr)} plays with {len(X_tr_sc[0]) if X_tr_sc else 0} players each...")
        model, loss = train_model_multi_player(
            X_tr_sc,
            y_tr_dx,
            y_tr_dy,
            mask_tr,
            X_va_sc,
            y_va_dx,
            y_va_dy,
            mask_va,
            input_dim,
        )

        # 训练集 RMSE
        print(f"  Computing train RMSE on {len(X_tr)} plays...")
        rmse_train = compute_val_rmse_multi_player_stt(
            model,
            X_tr_sc,
            y_tr_dx,
            y_tr_dy,
            mask_tr,
            Config.MAX_FUTURE_HORIZON,
            Config.DEVICE,
        )
        
        # 验证集 RMSE
        print(f"  Validating on {len(X_va)} plays...")
        rmse = compute_val_rmse_multi_player_stt(
            model,
            X_va_sc,
            y_va_dx,
            y_va_dy,
            mask_va,
            Config.MAX_FUTURE_HORIZON,
            Config.DEVICE,
        )

        print(
            f"[RESULT] seed {seed} fold {fold} → "
            f"Train RMSE={rmse_train:.4f} | "
            f"Val RMSE={rmse:.4f} | "
            f"Huber loss={loss:.5f}"
        )

        fold_rmses.append(rmse)
        all_rmse.append(rmse)
        cv_log.append(
            {
                "seed": seed,
                "fold": fold,
                "rmse": rmse,
                "loss": float(loss),
                "model_type": "multi_player",
            }
        )

        # Save model
        save_fold_artifacts_stt(
            seed=seed,
            fold=fold,
            scaler=scaler,
            model=model,
            base_dir=Config.SAVE_DIR,
        )
        print(f"  Model saved for fold {fold}")

    print(
        f"\n{'='*70}\n"
        f"[SEED SUMMARY] seed {seed} RMSEs: {[f'{r:.4f}' for r in fold_rmses]} | "
        f"mean={float(np.mean(fold_rmses)):.4f} yards\n"
        f"{'='*70}\n"
    )

    return all_rmse, cv_log


# ============================================================================
#                    多球员 GRU-Transformer 架构
# ============================================================================

class MultiPlayerGRUTransformer(nn.Module):
    """
    多球员空间-时间模型，用GRU处理每个球员的时序，用Transformer学习球员间交互
    
    Shape分析：
    ────────────────────────────────────────────────────────────────────
    输入:  (batch, n_players=22, seq_len, input_dim)
           示例: (32, 22, 10, 167)
    ────────────────────────────────────────────────────────────────────
    GRU处理 (seq_len维):
           输入 reshape:  (batch*22, seq_len, input_dim) = (704, 10, 167)
           GRU 输出:      (704, seq_len, hidden_dim) = (704, 10, 128)
           取最后帧:      (704, hidden_dim) = (704, 128)
           reshape回:     (batch, 22, hidden_dim) = (32, 22, 128)
    ────────────────────────────────────────────────────────────────────
    Transformer (player维):
           输入:  (batch, 22, hidden_dim) = (32, 22, 128)
           输出:  (batch, 22, hidden_dim) = (32, 22, 128)
    ────────────────────────────────────────────────────────────────────
    预测头 (Linear层):
           输入:  (batch, 22, hidden_dim) = (32, 22, 128)
           输出:  (batch, 22, 2*horizon) = (32, 22, 110)
    ────────────────────────────────────────────────────────────────────
    Reshape分离维度:
           (batch, 22, 2*horizon) → (batch, 22, horizon, 2)
           (32, 22, 110) → (32, 22, 55, 2)
    ────────────────────────────────────────────────────────────────────
    Cumsum (horizon维):
           输入:  (batch, 22, horizon, 2) = (32, 22, 55, 2)
           输出:  (batch, 22, horizon, 2) = (32, 22, 55, 2)
           [每个球员各自在horizon维度累积]
    ────────────────────────────────────────────────────────────────────
    """
    
    def __init__(self, input_dim: int, n_players: int = Config.MAX_NUM_PLAYER, dropout: float = 0.1):
        super().__init__()
        self.n_players = n_players
        self.horizon = Config.MAX_FUTURE_HORIZON
        self.hidden_dim = Config.HIDDEN_DIM
        
        # ⚠️ 用 TemporalTransformerEncoder 替换 GRU
        # 处理时序维度，在帧（horizon）维度上做 attention
        self.temporal_encoder = TemporalTransformerEncoder(
            input_dim=input_dim,
            dropout=dropout
        )
        
        # Transformer: 学习球员间交互（player 维度）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=Config.N_HEADS,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=Config.N_LAYERS
        )
        
        # LayerNorm: 在预测前稳定特征分布
        self.pred_ln = nn.LayerNorm(self.hidden_dim)
        
        # 预测头: 输出每个球员的 2*horizon 个值
        self.pred_head = nn.Linear(self.hidden_dim, 2 * self.horizon)
    
    def forward(self, x: torch.Tensor, tracked_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, 22, seq_len, input_dim)
               示例: (32, 22, 10, 167)
            tracked_mask: (batch, 22) - 可选，1表示被追踪/有效，0表示未追踪/pad
                          示例: (32, 22)
        
        Returns:
            output: (batch, 22, horizon, 2)
                    示例: (32, 22, 55, 2)
        """
        batch_size, n_players, seq_len, input_dim = x.shape
        
        # ============ Step 1: 时序Transformer处理 ============
        # reshape: (batch, 22, seq_len, input_dim) -> (batch*22, seq_len, input_dim)
        x_flat = x.view(batch_size * n_players, seq_len, input_dim)
        # print(f"[Temporal Encoder Input] {x_flat.shape}")  # (704, 10, 167)
        
        # ⚠️ 用 TemporalTransformerEncoder 替换 GRU
        # 在时序维度上做 Transformer + Attention pooling
        h = self.temporal_encoder(x_flat)  # (batch*22, hidden_dim)
        # print(f"[Temporal Encoder Output] {h.shape}")  # (704, 128)
        
        # reshape回: (batch*22, hidden_dim) -> (batch, 22, hidden_dim)
        h = h.view(batch_size, n_players, self.hidden_dim)
        # print(f"[After Reshape] {h.shape}")  # (32, 22, 128)
        
        # ============ Step 2: Transformer学习球员间交互 ============
        # 输入: (batch, 22, hidden_dim) -> 输出: (batch, 22, hidden_dim)
        # 创建球员维度的attention mask：True表示要屏蔽的位置
        src_key_padding_mask = None
        if tracked_mask is not None:
            src_key_padding_mask = (tracked_mask == 0)  # (batch, 22)
            # print(f"[Key Padding Mask] {src_key_padding_mask.shape}")  # (32, 22)
        
        attn_out = self.transformer_encoder(
            h,
            src_key_padding_mask=src_key_padding_mask
        )
        # print(f"[Transformer Output] {attn_out.shape}")  # (32, 22, 128)
        
        # ============ Step 3: 预测头 ============
        
        # ⚠️ Reshape 保证各球员独立处理
        # (batch, 22, hidden_dim) -> (batch*22, hidden_dim)
        attn_out_flat = attn_out.view(batch_size * n_players, self.hidden_dim)

        # 输入: (batch*22, hidden_dim) -> 输出: (batch*22, hidden_dim)
        # 先进行 LayerNorm 稳定特征
        attn_out = self.pred_ln(attn_out)
        
        # 预测
        pred_flat = self.pred_head(attn_out_flat)  # (batch*22, 2*horizon)
        
        # ============ Step 4: 恢复形状并分离 dx/dy 和 horizon ============
        # (batch*22, 2*horizon) -> (batch, 22, horizon, 2)
        pred = pred_flat.view(batch_size, n_players, self.horizon, 2)
        # print(f"[After Reshape] {pred.shape}")  # (32, 22, 55, 2)
        
        # ============ Step 5: Cumsum (每个球员各自在horizon维度) ============
        # 在 dim=2 (horizon维度) 累积
        output = torch.cumsum(pred, dim=2)
        # print(f"[Final Output] {output.shape}")  # (32, 22, 55, 2)
        
        return output


def prepare_multi_player_targets(
    batch_players_dx, batch_players_dy, max_h, n_players=Config.MAX_NUM_PLAYER
):
    """
    为多球员模型准备目标
    
    Args:
        batch_players_dx: List[(n_players, horizon)] - 各 horizon 长度可能不同
        batch_players_dy: List[(n_players, horizon)] - 各 horizon 长度可能不同
        max_h: 最大horizon（用于填充）
        n_players: 球员数
    
    Returns:
        targets: (batch, 22, max_h, 2)
        masks: (batch, 22, max_h)
    """
    batch_targets = []
    batch_masks = []
    
    for players_dx, players_dy in zip(batch_players_dx, batch_players_dy):
        # players_dx: (n_players, horizon_i)  - 长度可能不同
        # players_dy: (n_players, horizon_i)
        
        # 获取实际的 horizon 长度
        actual_horizon = players_dx.shape[1]
        
        # 如果长度不足 max_h，进行填充
        if actual_horizon < max_h:
            pad_len = max_h - actual_horizon
            players_dx = np.pad(players_dx, ((0, 0), (0, pad_len)), mode='constant', constant_values=0.0)
            players_dy = np.pad(players_dy, ((0, 0), (0, pad_len)), mode='constant', constant_values=0.0)
        
        # 转换成tensor
        dx_t = torch.tensor(players_dx[:, :max_h], dtype=torch.float32)
        dy_t = torch.tensor(players_dy[:, :max_h], dtype=torch.float32)
        
        # stack: (n_players, max_h, 2)
        player_target = torch.stack([dx_t, dy_t], dim=-1)
        batch_targets.append(player_target)
        
        # mask: (n_players, max_h) - 全1表示所有位置都有效
        mask = torch.ones(n_players, max_h, dtype=torch.float32)
        batch_masks.append(mask)
    
    # 输出: (batch, 22, max_h, 2) 和 (batch, 22, max_h)
    return torch.stack(batch_targets), torch.stack(batch_masks)


def _build_multi_player_train_batches(X_train_multi, y_train_multi_dx, y_train_multi_dy, player_masks_multi, shuffle=True):
    """
    构建多球员训练批次
    
    Args:
        X_train_multi: List of (22, seq_len, n_features)
        y_train_multi_dx: List of (22, horizon)
        y_train_multi_dy: List of (22, horizon)
        player_masks_multi: List of (22, horizon) - 球员有效性mask
    """
    indices = np.arange(len(X_train_multi))
    if shuffle:
        np.random.shuffle(indices)
    
    train_batches = []
    for i in range(0, len(X_train_multi), Config.BATCH_SIZE):
        end = min(i + Config.BATCH_SIZE, len(X_train_multi))
        batch_indices = indices[i:end]
        
        X_batch = [X_train_multi[j] for j in batch_indices]
        y_dx_batch = [y_train_multi_dx[j] for j in batch_indices]
        y_dy_batch = [y_train_multi_dy[j] for j in batch_indices]
        masks_batch = [player_masks_multi[j] for j in batch_indices]
        
        # Stack: (batch, 22, seq_len, n_features)
        bx = torch.tensor(np.stack(X_batch).astype(np.float32))
        by, bm = prepare_multi_player_targets(y_dx_batch, y_dy_batch, Config.MAX_FUTURE_HORIZON)
        
        # 从 player_masks 中提取 tracked_mask
        # tracked_mask: (batch, 22) - 1表示该球员至少有一个有效的horizon，0表示全为无效
        btm_list = []
        for mask in masks_batch:  # mask: (22, horizon)
            # 如果某个球员在任何horizon位置都有效(mask>0)，则该球员被追踪
            tracked = (mask.sum(axis=1) > 0).astype(np.float32)
            btm_list.append(tracked)
        btm = torch.tensor(np.stack(btm_list), dtype=torch.float32)
        
        train_batches.append((bx, by, bm, btm))
    
    return train_batches


def _build_multi_player_val_batches(X_val_multi, y_val_multi_dx, y_val_multi_dy, player_masks_multi):
    """
    构建多球员验证批次
    
    Args:
        X_val_multi: List of (22, seq_len, n_features)
        y_val_multi_dx: List of (22, horizon)
        y_val_multi_dy: List of (22, horizon)
        player_masks_multi: List of (22, horizon) - 球员有效性mask
    """
    val_batches = []
    for i in range(0, len(X_val_multi), Config.BATCH_SIZE):
        end = min(i + Config.BATCH_SIZE, len(X_val_multi))
        
        X_batch = [X_val_multi[j] for j in range(i, end)]
        y_dx_batch = [y_val_multi_dx[j] for j in range(i, end)]
        y_dy_batch = [y_val_multi_dy[j] for j in range(i, end)]
        masks_batch = [player_masks_multi[j] for j in range(i, end)]
        
        bx = torch.tensor(np.stack(X_batch).astype(np.float32))
        by, bm = prepare_multi_player_targets(y_dx_batch, y_dy_batch, Config.MAX_FUTURE_HORIZON)
        
        # 从 player_masks 中提取 tracked_mask
        btm_list = []
        for mask in masks_batch:  # mask: (22, horizon)
            tracked = (mask.sum(axis=1) > 0).astype(np.float32)
            btm_list.append(tracked)
        btm = torch.tensor(np.stack(btm_list), dtype=torch.float32)
        
        val_batches.append((bx, by, bm, btm))
    
    return val_batches


def criterion_multi_player(pred, target, player_mask, tracked_mask):
    """
    多球员损失函数
    
    Args:
        pred: (batch, 22, horizon, 2) - 模型预测
        target: (batch, 22, horizon, 2) - 真实轨迹
        player_mask: (batch, 22, horizon) - 有效帧mask
        tracked_mask: (batch, 22) - 被追踪球员mask
    
    Returns:
        scalar loss
    """
    # 计算误差
    err = pred - target
    abs_err = torch.abs(err)
    
    # Huber损失
    delta = 0.5
    huber = torch.where(
        abs_err <= delta,
        0.5 * err * err,
        delta * (abs_err - 0.5 * delta),
    )
    
    # 时间衰减权重
    horizon = pred.size(2)
    t = torch.arange(horizon, device=pred.device, dtype=pred.dtype)
    w_time = torch.exp(-0.03 * t).view(1, 1, -1, 1)
    huber = huber * w_time
    
    # 应用player mask (只计算被追踪的球员)
    tracked_mask = tracked_mask.unsqueeze(-1).unsqueeze(-1)  # (batch, 22, 1, 1)
    player_mask = player_mask.unsqueeze(-1)  # (batch, 22, horizon, 1)
    
    final_mask = tracked_mask * player_mask * w_time
    
    # 计算损失
    loss = (huber * final_mask).sum() / (final_mask.sum() + 1e-8)
    
    return loss


def train_model_multi_player(
    X_train_multi,
    y_train_multi_dx,
    y_train_multi_dy,
    player_masks_train,
    X_val_multi,
    y_val_multi_dx,
    y_val_multi_dy,
    player_masks_val,
    input_dim,
):
    """
    训练多球员模型
    
    Args:
        X_train_multi: List of (22, seq_len, input_dim)
        y_train_multi_dx: List of (22, horizon)
        y_train_multi_dy: List of (22, horizon)
        player_masks_train: List of (22, horizon) - 训练集球员mask
        X_val_multi: List of (22, seq_len, input_dim)
        y_val_multi_dx: List of (22, horizon)
        y_val_multi_dy: List of (22, horizon)
        player_masks_val: List of (22, horizon) - 验证集球员mask
        input_dim: 特征维度
    """
    device = Config.DEVICE
    
    # 构建验证批次
    val_batches = _build_multi_player_val_batches(
        X_val_multi, y_val_multi_dx, y_val_multi_dy, player_masks_val
    )
    
    # 定义模型
    model = MultiPlayerGRUTransformer(
        input_dim=input_dim,
        n_players=Config.MAX_NUM_PLAYER,
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5
    )
    
    best_loss, best_state, bad = float("inf"), None, 0
    start_time = time.time()
    
    for epoch in range(1, Config.EPOCHS + 1):
        # 构建训练批次
        train_batches = _build_multi_player_train_batches(
            X_train_multi, y_train_multi_dx, y_train_multi_dy, player_masks_train, shuffle=True
        )
        
        model.train()
        train_losses = []
        for bx, by, bm, btm in train_batches:
            bx, by, bm, btm = bx.to(device), by.to(device), bm.to(device), btm.to(device)
            pred = model(bx, tracked_mask=btm)
            
            loss = criterion_multi_player(pred, by, bm, btm)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by, bm, btm in val_batches:
                bx, by, bm, btm = bx.to(device), by.to(device), bm.to(device), btm.to(device)
                pred = model(bx, tracked_mask=btm)
                
                val_loss = criterion_multi_player(pred, by, bm, btm)
                val_losses.append(val_loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            total_time = time.time() - start_time
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)
            print(
                f"  Epoch {epoch:>3}: train={train_loss:.4f}, val={val_loss:.4f}, "
                f"Time_elapsed={minutes:>2}min {seconds:>2}s"
            )
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= Config.PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, best_loss
