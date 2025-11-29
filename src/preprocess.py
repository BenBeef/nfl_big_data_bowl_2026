import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Queue

from .config import Config
from .utils import (
    build_play_direction_map,
    unify_left_direction_ipt,
    unify_left_direction_opt,
)
from .feature import FeatureEngineer


def _canonicalize_key_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("game_id", "play_id", "nfl_id"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Handle missing keys
    df = df.dropna(subset=["game_id", "play_id", "nfl_id"])
    # Convert to int64?
    df["game_id"] = df["game_id"].astype("int64")
    df["play_id"] = df["play_id"].astype("int64")
    df["nfl_id"] = df["nfl_id"].astype("int64")
    return df


def _process_group_batch(
    batch_keys: list,
    grouped_dict: dict,
    feature_cols: list,
    target_rows: pd.DataFrame,
    idx_x: int,
    idx_y: int,
    dir_map: pd.DataFrame,
    queue: Queue,
):
    sequences, targets_dx, targets_dy, targets_fids, seq_meta = [], [], [], [], []
    for key in batch_keys:
        gid, pid, nid = key
        group_df = grouped_dict.get(key)
        if group_df is None:
            continue

        # Build input window
        input_window = group_df.tail(Config.WINDOW_SIZE)
        if len(input_window) < Config.WINDOW_SIZE:
            pad_len = Config.WINDOW_SIZE - len(input_window)
            pad_df = pd.DataFrame(
                np.nan, index=range(pad_len), columns=input_window.columns
            )
            input_window = pd.concat([pad_df, input_window], ignore_index=True)

        input_window = input_window.fillna(input_window.mean(numeric_only=True))
        seq = input_window[feature_cols].to_numpy(dtype=np.float32)
        seq = np.nan_to_num(seq, nan=0.0)
        sequences.append(seq)

        # Training targets
        if Config.TRAIN:
            out_grp: pd.DataFrame = target_rows[
                (target_rows["game_id"] == gid)
                & (target_rows["play_id"] == pid)
                & (target_rows["nfl_id"] == nid)
            ].sort_values("frame_id")
            if len(out_grp) == 0:
                sequences.pop()
                continue
            dx = out_grp["x"].to_numpy(np.float32) - seq[-1, idx_x]
            dy = out_grp["y"].to_numpy(np.float32) - seq[-1, idx_y]
            fids = out_grp["frame_id"].to_numpy(np.int32)
            targets_dx.append(dx)
            targets_dy.append(dy)
            targets_fids.append(fids)

        play_dir_val = dir_map.loc[(gid, pid)]
        seq_meta.append(
            {
                "game_id": gid,
                "play_id": pid,
                "nfl_id": nid,
                "frame_id": int(input_window.iloc[-1]["frame_id"]),
                "play_direction": play_dir_val,
            }
        )

        if queue is not None:
            queue.put(1)

    return sequences, targets_dx, targets_dy, targets_fids, seq_meta


def _convert_to_multi_player_format(
    sequences: list,
    targets_dx: list,
    targets_dy: list,
    seq_meta: list,
    seq_len: int,
    n_features: int,
    n_players: int = Config.MAX_NUM_PLAYER,
):
    """
    将单个球员格式的序列转换为多球员格式 (play, n_players, seq_len, n_features)
    
    Args:
        sequences: List of (seq_len, n_features) - 所有球员的序列
        targets_dx: List of (horizon,) - 位移目标 (x方向)
        targets_dy: List of (horizon,) - 位移目标 (y方向)
        seq_meta: List of dict - 序列元数据
        seq_len: 序列长度
        n_features: 特征维度
        n_players: 每个play的球员数 (默认22)
    
    Returns:
        sequences_multi: List of (n_players, seq_len, n_features)
        targets_dx_multi: List of (n_players, Config.MAX_FUTURE_HORIZON)
        targets_dy_multi: List of (n_players, Config.MAX_FUTURE_HORIZON)
        seq_meta_multi: List of dict - play级别的元数据
    """
    # 直接使用 Config 中的最大 horizon
    max_horizon = Config.MAX_FUTURE_HORIZON
    
    # 按 (game_id, play_id) 分组
    play_groups = {}  # key: (game_id, play_id), value: list of indices
    
    for idx, meta in enumerate(seq_meta):
        play_key = (meta["game_id"], meta["play_id"])
        if play_key not in play_groups:
            play_groups[play_key] = []
        play_groups[play_key].append(idx)
    
    sequences_multi = []
    targets_dx_multi = []
    targets_dy_multi = []
    seq_meta_multi = []
    
    for (gid, pid), player_indices in play_groups.items():
        # 该 play 中的球员数
        n_actual_players = len(player_indices)
        
        # 初始化多球员序列 (n_players, seq_len, n_features)
        # 使用 n_players 来标准化，不足的用零填充
        play_seqs = np.zeros((n_players, seq_len, n_features), dtype=np.float32)
        play_targets_dx = np.zeros((n_players, max_horizon), dtype=np.float32)
        play_targets_dy = np.zeros((n_players, max_horizon), dtype=np.float32)
        
        # 填充实际的球员数据
        for player_slot, idx in enumerate(player_indices):
            if player_slot < n_players:
                play_seqs[player_slot] = sequences[idx]
                if targets_dx and idx < len(targets_dx):
                    dx_val = targets_dx[idx]
                    # 填充到 max_horizon 长度 (不足的用0填充)
                    play_targets_dx[player_slot, :len(dx_val)] = dx_val
                if targets_dy and idx < len(targets_dy):
                    dy_val = targets_dy[idx]
                    # 填充到 max_horizon 长度 (不足的用0填充)
                    play_targets_dy[player_slot, :len(dy_val)] = dy_val
        
        sequences_multi.append(play_seqs)
        if targets_dx:
            targets_dx_multi.append(play_targets_dx)
        if targets_dy:
            targets_dy_multi.append(play_targets_dy)
        
        # Play级别的元数据 (取第一个球员的信息)
        first_meta = seq_meta[player_indices[0]]
        play_meta = {
            "game_id": gid,
            "play_id": pid,
            "frame_id": first_meta["frame_id"],
            "play_direction": first_meta["play_direction"],
            "n_players": n_actual_players,
        }
        seq_meta_multi.append(play_meta)
    
    return sequences_multi, targets_dx_multi, targets_dy_multi, seq_meta_multi


def prepare_sequences_with_advanced_features(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    feature_groups: list = None,
):

    print(f"\n{'='*80}")
    print(f"PREPARING SEQUENCES WITH ADVANCED FEATURES (MULTI-PLAYER FORMAT)")
    print(f"{'='*80}")
    print(f"Window size: {Config.WINDOW_SIZE}")

    # game_id/play_id/nfl_id 字段类型强制转换
    input_df = _canonicalize_key_dtypes(input_df)
    output_df = _canonicalize_key_dtypes(output_df)

    # 去重 game_id, play_id, play_direction
    dir_map = build_play_direction_map(input_df)
    # 统一输入数据方向为向左
    input_df = unify_left_direction_ipt(input_df)
    # 统一输出数据的方向为左
    output_df = unify_left_direction_opt(output_df, dir_map)

    target_rows = output_df
    target_groups = output_df[["game_id", "play_id", "nfl_id"]].drop_duplicates()

    # Feature Engineering
    fe = FeatureEngineer(feature_groups)
    processed_df, feature_cols = fe.transform(input_df)

    # Build sequences
    start_time = time.time()
    grouped_dict = {
        (gid, pid, nid): g
        for (gid, pid, nid), g in processed_df.groupby(
            ["game_id", "play_id", "nfl_id"], sort=False
        )
    }

    # helpful indices
    idx_x = feature_cols.index("x")
    idx_y = feature_cols.index("y")

    # Spread group across cpus
    all_keys = [tuple(x) for x in target_groups.to_numpy()]
    batch_size = (len(all_keys) + Config.MAX_WORKER - 1) // Config.MAX_WORKER
    batches = [
        all_keys[i : i + batch_size] for i in range(0, len(all_keys), batch_size)
    ]

    sequences, targets_dx, targets_dy, targets_fids, seq_meta = [], [], [], [], []

    if Config.TRAIN:
        manager = Manager()
        queue = manager.Queue()
        pbar = tqdm(total=len(all_keys), desc="Creating sequences (groups)")

        # Build sequences in parallel
        with ProcessPoolExecutor(max_workers=Config.MAX_WORKER) as ex:
            futures = [
                ex.submit(
                    _process_group_batch,
                    b,
                    grouped_dict,
                    feature_cols,
                    target_rows,
                    idx_x,
                    idx_y,
                    dir_map,
                    queue,
                )
                for b in batches
            ]
            finished = 0
            while finished < len(all_keys):
                queue.get()
                finished += 1
                pbar.update(1)

            # Wait for all task to complete
            for fut in as_completed(futures):
                seqs, dxs, dys, fids_list, metas = fut.result()
                sequences.extend(seqs)
                targets_dx.extend(dxs)
                targets_dy.extend(dys)
                targets_fids.extend(fids_list)
                seq_meta.extend(metas)

        pbar.close()

    else:
        # No multiprocessing when not training
        print("[INFO] Running in single-process mode")
        pbar = tqdm(total=len(all_keys), desc="Creating sequences (groups)")
        for key in all_keys:
            seqs, dxs, dys, fids_list, metas = _process_group_batch(
                [key],
                grouped_dict,
                feature_cols,
                target_rows,
                idx_x,
                idx_y,
                dir_map,
                None,
            )
            sequences.extend(seqs)
            seq_meta.extend(metas)
            pbar.update(1)
        pbar.close()
    end_time = time.time()
    print(f"Created {len(sequences)} sequences with {len(feature_cols)} features each")
    print(f"Time to build sequences: {end_time - start_time:.2f} seconds")

    # ========================================================================
    # 转换为多球员格式: (play, 22, seq_len, input_dim)
    # ========================================================================
    print(f"\nConverting to multi-player format (for MultiPlayerGRUTransformer)...")
    
    sequences_multi, targets_dx_multi, targets_dy_multi, seq_meta_multi = _convert_to_multi_player_format(
        sequences, 
        targets_dx, 
        targets_dy, 
        seq_meta,
        Config.WINDOW_SIZE,
        len(feature_cols)
    )
    
    print(f"Multi-player sequences: {len(sequences_multi)} plays, each with 22 players")
    if len(sequences_multi) > 0:
        print(f"  Shape per play: (22, {Config.WINDOW_SIZE}, {len(feature_cols)})")
        if targets_dx_multi:
            print(f"  Target shapes: (22, {targets_dx_multi[0].shape[1]})")

    if Config.TRAIN:
        return (
            sequences_multi,
            targets_dx_multi,
            targets_dy_multi,
            targets_fids,
            seq_meta_multi,
            feature_cols,
        )
    return sequences_multi, seq_meta_multi, feature_cols
