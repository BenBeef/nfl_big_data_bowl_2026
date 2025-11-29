import numpy as np 
from sklearn.model_selection import GroupKFold
from src.model import train_all_folds_multi_player_stt
from src.config import Config
from src.utils import set_seed, load_input_output, write_meta
from src.preprocess import prepare_sequences_with_advanced_features
from datetime import datetime

timestamp = lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S')


if __name__ == '__main__':

    print(f"[1/4] [{timestamp()}] Load Training data...")
    train_input, train_output = load_input_output()

    feature_groups = [
         "target_alignment",
        # "multi_window",
         "multi_window",
        "lag",
        "motion_change",
        "field_position",
        "distance_rate",
        "geometric",
        "neighbor_gnn",
        "time",
        # "role",
        "role",
        "passer",
        "curvature",
        "route",
        "receiver",
    ]

    print(f"\n[2/4] [{timestamp()}] Prepare sequences data...")
    result = prepare_sequences_with_advanced_features(train_input, train_output, feature_groups)
    sequences, targets_dx, targets_dy, targets_fids, seq_meta, feature_cols = result

    # 生成 player_mask: (n_plays, 22, horizon)
    # 基于 targets_dx/dy 是否非零来确定球员是否被追踪
    print(f"\n[2.5/4] [{timestamp()}] Creating player masks...")
    player_masks = []
    for dx, dy in zip(targets_dx, targets_dy):
        # dx, dy: (22, horizon)
        # mask: 1 如果该球员的任何目标不为零，否则 0
        player_mask = np.zeros((dx.shape[0], dx.shape[1]), dtype=np.float32)
        for player_idx in range(dx.shape[0]):
            # 检查该球员是否有有效的目标 (不全为零)
            if np.any(dx[player_idx] != 0) or np.any(dy[player_idx] != 0):
                player_mask[player_idx, :] = 1.0
        player_masks.append(player_mask)
    
    print(f"Created player masks for {len(player_masks)} plays")

    
    print(f"\n[3/4] [{timestamp()}] Training all folds (Multi-Player Model)...")
    
    input_dim = len(feature_cols)
    seed = Config.SEEDS[0]
    gkf = GroupKFold(n_splits=Config.N_FOLDS)
    groups = np.array([d['game_id'] for d in seq_meta])

    train_all_folds_multi_player_stt(
        gkf, sequences, groups, targets_dx, targets_dy, player_masks, seed, input_dim
    )


    print(f"\n[4/4] [{timestamp()}] Save Meta data...")
    write_meta(
        feature_cols = feature_cols, 
        base_dir=Config.SAVE_DIR,
        feature_groups = feature_groups,
        save_src=True
    )

    print(f"\n[{timestamp()}] finished..")