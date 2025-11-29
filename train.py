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
    sequences, targets_dx, targets_dy, targets_fids, seq_meta, player_masks, feature_cols = result
    
    print(f"  Generated player masks for {len(player_masks)} plays")

    print(f"\n[3/4] [{timestamp()}] Training all folds (Multi-Player Model)...")
    
    input_dim = len(feature_cols)
    seed = Config.SEEDS[0]
    set_seed(seed)
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