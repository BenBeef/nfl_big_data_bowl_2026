import numpy as np
from .model import train_all_folds_stt
from .preprocess import prepare_sequences_with_advanced_features
from .config import Config
from sklearn.model_selection import GroupKFold
from datetime import datetime
import pandas as pd


fine_cnt = 19 if not Config.DEBUG else 2
timestamp = lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f"[1/4] [{timestamp()}] Loading {fine_cnt} files for training...")
train_input_files = [Config.DATA_DIR / f"train/input_2023_w{w:02d}.csv" for w in range(1, fine_cnt)]
train_output_files = [Config.DATA_DIR / f"train/output_2023_w{w:02d}.csv" for w in range(1, fine_cnt)]
train_input = pd.concat([pd.read_csv(f) for f in train_input_files if f.exists()])
train_output = pd.concat([pd.read_csv(f) for f in train_output_files if f.exists()])

sequences, targets_dx, targets_dy, targets_fids, seq_meta, feature_cols = prepare_sequences_with_advanced_features(
    train_input,
    train_output,
    Config.FEATURE_GROUPS,
)

# gkf, sequences, groups, targets_dx, targets_dy, seed, input_dim
gkf = GroupKFold(n_splits=Config.N_FOLDS)
groups = np.array([d['game_id'] for d in seq_meta])
gkf = GroupKFold(n_splits=Config.N_FOLDS)
seed = Config.SEEDS[0]
input_dim = Config.HIDDEN_DIM
train_all_folds_stt(gkf, sequences, groups, targets_dx, targets_dy, seed, input_dim)
