from ast import List
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


def _process_no_predict_batch(
    batch_keys: list,
    grouped_no_predict_dict: dict,
    feature_cols: list,
    seq_len: int,
    queue = None
):
    """⭐ 多进程处理非预测球员数据"""
    not_predict_groups = {}
    for (gid, pid) in batch_keys:
        play_nfl_groups = {}
        play_keys = [k for k in grouped_no_predict_dict.keys() if k[:2] == (gid, pid)]
        for key in play_keys:
            group = grouped_no_predict_dict[key]
            nfl_id = key[2]
            group = group.sort_values("frame_id")
            input_window = group[feature_cols].tail(seq_len)
            if len(input_window) < seq_len:
                pad_len = seq_len - len(input_window)
                pad_df = pd.DataFrame(np.nan, index=range(pad_len), columns=input_window.columns)
                input_window = pd.concat([pad_df, input_window], ignore_index=True)
            input_window = input_window.fillna(input_window.mean(numeric_only=True))
            play_nfl_groups[nfl_id] = input_window[feature_cols].to_numpy(dtype=np.float32)
        if play_nfl_groups:
            not_predict_groups[(gid, pid)] = play_nfl_groups
        if queue is not None:
            queue.put(1)
    return not_predict_groups


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


def add_relative_feature(f_idx, play_seqs, n_filled_player, n_players, seq_len, def_val=0.0, fill_val=-0.0):
    play_rel_features = np.full((n_players, seq_len, n_players), fill_value=fill_val, dtype=np.float32)
    # 提取位置信息 (使用 f_idx)
    pos_val = play_seqs[..., f_idx]  # (n_players, seq_len)
    
    # 向量化操作：计算相对值
    # positions_x: (n_players, seq_len)
    # 使用 broadcasting 计算所有对间的差值
    positions_v_i = pos_val[:n_filled_player, np.newaxis, :]  # (next_slot, 1, seq_len)
    positions_v_j = pos_val[np.newaxis, :n_filled_player, :]  # (1, next_slot, seq_len)
    
    rel_x = positions_v_i - positions_v_j  # (next_slot, next_slot, seq_len)
    
    # 填充到 play_rel_features
    for i in range(n_filled_player):
        for j in range(n_filled_player):
            if i != j:
                play_rel_features[i, :, j] = rel_x[i, j, :]
            else:
                # 自己对自己为 0（已初始化）
                play_rel_features[i, :, j] = def_val
    return play_rel_features
    

def _convert_to_multi_player_format(
    sequences: list,
    targets_dx: list,
    targets_dy: list,
    seq_meta: list,
    seq_len: int,
    feature_cols:list,
    df_no_predicted: pd.DataFrame = None,
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
    player_masks_multi = []
    seq_meta_multi = []
    rel_features_multi = []  # ⭐ 相对位移特征

    max_player_slot = -1

    f_x_idx, f_y_idx = feature_cols.index('x'), feature_cols.index('y')
    f_velocity_x_idx, f_velocity_y_idx = feature_cols.index('velocity_x'), feature_cols.index('velocity_y')
    dir_idx = feature_cols.index('dir')
    n_features = len(feature_cols)
    n_features = len(feature_cols)
    
    # ⭐ 多进程处理非预测球员数据（如果提供了）
    not_predict_groups = {}
    if df_no_predicted is not None:
        # 预先分组 df_no_predicted
        grouped_no_predict_dict = {
            (gid, pid, nfl_id): group
            for (gid, pid, nfl_id), group in df_no_predicted.groupby(
                ["game_id", "play_id", "nfl_id"], sort=False
            )
        }
        
        # 获取所有 (gid, pid) 对
        play_keys = sorted(set((k[0], k[1]) for k in grouped_no_predict_dict.keys()))
        
        # 分批处理
        batch_size = (len(play_keys) + Config.MAX_WORKER - 1) // Config.MAX_WORKER
        batches = [play_keys[i:i+batch_size] for i in range(0, len(play_keys), batch_size)]
        
        grouped_no_predict_dicts = [{} for _ in range(len(batches))]
        game_dict = {}
        for key in grouped_no_predict_dict:
            gid, pid, nfl_id = key
            game_dict[(gid, pid)] = len(game_dict) % len(grouped_no_predict_dicts)

        for key, val in grouped_no_predict_dict.items():
            gid, pid, nfl_id = key
            idx = game_dict[(gid, pid)]
            grouped_no_predict_dicts[idx][key] = val.copy()
        
        print(f'Processing no predict players...')
        # 多进程处理
        pbar = tqdm(total=len(play_keys), desc="Processing non-predicted players")
        manager = Manager()
        queue = manager.Queue()
        
        with ProcessPoolExecutor(max_workers=Config.MAX_WORKER) as ex:
            futures = [
                ex.submit(
                    _process_no_predict_batch, 
                    b, 
                    grouped_no_predict_dicts[i], 
                    feature_cols, 
                    seq_len,
                    queue 
                    )
                for i, b in enumerate(batches)
            ]

            finished = 0
            while finished < len(play_keys):
                queue.get()
                finished += 1
                pbar.update(1)
            
            # 按完成顺序收集结果
            for fut in as_completed(futures):
                batch_result = fut.result()
                not_predict_groups.update(batch_result)
        
        pbar.close()
    
    for (gid, pid), player_indices in play_groups.items():
        # 该 play 中的球员数
        n_actual_players = len(player_indices)
        
        # 初始化多球员序列 (n_players, seq_len, n_features)
        # 使用 n_players 来标准化，不足的用零填充
        play_seqs = np.zeros((n_players, seq_len, n_features), dtype=np.float32)
        play_targets_dx = np.zeros((n_players, max_horizon), dtype=np.float32)
        play_targets_dy = np.zeros((n_players, max_horizon), dtype=np.float32)

        max_player_slot = max(max_player_slot, len(player_indices))
        
        # 初始化 mask
        play_mask = np.zeros((n_players, max_horizon), dtype=np.float32)
        
        # 填充实际的球员数据和生成 mask
        for player_slot, idx in enumerate(player_indices):
            if player_slot < n_players:
                play_seqs[player_slot] = sequences[idx]
                
                # 填充值和mask
                if targets_dx and idx < len(targets_dx) and len(targets_dx[idx]) > 0:
                    dx_val = targets_dx[idx]
                    play_targets_dx[player_slot, :len(dx_val)] = dx_val
                    # dy值填充
                    dy_val = targets_dy[idx]
                    play_targets_dy[player_slot, :len(dy_val)] = dy_val
                    # mask值
                    play_mask[player_slot, :len(dx_val)] = 1.0
        
        # ⭐ 初始化 next_slot，用于填充非预测球员
        next_slot = len(player_indices)
        
        # ⭐ 填充非预测球员（如果提供了）
        if (gid, pid) in not_predict_groups:
            play_nfl_groups = not_predict_groups[(gid, pid)]
            
            # 填充未预测的球员
            for nfl_id in play_nfl_groups:
                # ⭐ 直接获取预处理好的数据（已经过 tail + pad + fillna）
                player_seq = play_nfl_groups[nfl_id]
                if len(player_seq) == 0:
                    continue
                
                # 处理 NaN
                player_seq = np.nan_to_num(player_seq, nan=0.0)
                
                # 填充到 play_seqs
                play_seqs[next_slot] = player_seq
                next_slot += 1
        
        # ⭐ 计算相对位移特征（填充完play_seqs后立即计算）
        # 初始化相对位移特征 (2 * n_players: x和y各n_players列)
        # play_rel_features = np.full((n_players, seq_len, 2 * n_players), fill_value=-300, dtype=np.float32)
        # # 提取位置信息 (使用 f_x_idx, f_y_idx)
        # positions_x = play_seqs[..., f_x_idx]  # (n_players, seq_len)
        # positions_y = play_seqs[..., f_y_idx]  # (n_players, seq_len)
        
        # ⭐ 计算所有球员间的相对位移（包括非预测球员）
        # n_filled_players = next_slot
        
        # # ⭐ 向量化操作：计算相对位移
        # # positions_x: (n_players, seq_len)
        # # 使用 broadcasting 计算所有对间的差值
        # positions_x_i = positions_x[:n_filled_players, np.newaxis, :]  # (n_filled_players, 1, seq_len)
        # positions_x_j = positions_x[np.newaxis, :n_filled_players, :]  # (1, n_filled_players, seq_len)
        # positions_y_i = positions_y[:n_filled_players, np.newaxis, :]  # (n_filled_players, 1, seq_len)
        # positions_y_j = positions_y[np.newaxis, :n_filled_players, :]  # (1, n_filled_players, seq_len)
        
        # # 计算相对位移 (n_filled_players, n_filled_players, seq_len)
        # rel_x = positions_x_j - positions_x_i  # (n_filled_players, n_filled_players, seq_len)
        # rel_y = positions_y_j - positions_y_i  # (n_filled_players, n_filled_players, seq_len)
        
        # # 填充到 play_rel_features
        # for i in range(n_filled_players):
        #     for j in range(n_filled_players):
        #         if i != j:
        #             play_rel_features[i, :, j] = rel_x[i, j, :]
        #             play_rel_features[i, :, n_players + j] = rel_y[i, j, :]
        #         else:
        #             # 自己对自己为 0（已初始化）
        #             play_rel_features[i, :, j] = 0.0
        #             play_rel_features[i, :, n_players + j] = 0.0

        # 计算相对特征
        # 相对位移
        x_rel_feat = add_relative_feature(f_x_idx, play_seqs, next_slot, n_players, seq_len)
        y_rel_feat = add_relative_feature(f_y_idx, play_seqs, next_slot, n_players, seq_len)
        # 相对距离
        dis_rel_feat = np.sqrt(x_rel_feat ** 2 + y_rel_feat ** 2)

        vel_x_rel_feat = add_relative_feature(f_velocity_x_idx, play_seqs, next_slot, n_players, seq_len)
        vel_y_rel_feat = add_relative_feature(f_velocity_y_idx, play_seqs, next_slot, n_players, seq_len)

        dir_rel_feat = add_relative_feature(dir_idx, play_seqs, next_slot, n_players, seq_len)

        rel_features = [x_rel_feat, y_rel_feat, dis_rel_feat, vel_x_rel_feat, vel_y_rel_feat, dir_rel_feat]

        # 只保留相对12码之内的相对属性
        # dis_mask = dis_rel_feature <= 12.0
        # rel_features = [fe * dis_mask for fe in rel_features]  # 只保存某个

        # 合并相对特征（沿最后一个维度连接）
        play_rel_features = np.concatenate(rel_features, axis=-1, dtype=np.float32)
        # shape: (n_players, seq_len, 2*n_players)
        
        sequences_multi.append(play_seqs)
        # print("positions_x\n", positions_x)
        # print("positions_y\n", positions_y)
        # print("play_rel_features\n", play_rel_features)
        if targets_dx:
            targets_dx_multi.append(play_targets_dx)
        if targets_dy:
            targets_dy_multi.append(play_targets_dy)
        player_masks_multi.append(play_mask)
        rel_features_multi.append(play_rel_features)  # ⭐ 添加相对位移特征
        
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
    
    print(f'Max predicted players per play, max_player_slot= {max_player_slot}')
    
    return sequences_multi, targets_dx_multi, targets_dy_multi, player_masks_multi, seq_meta_multi, rel_features_multi


def prepare_sequences_with_advanced_features(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    feature_groups: list = None,
    multi_player=True
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
    processed_df, feature_cols, df_no_predicted = fe.transform(input_df)

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
    if multi_player:
        sequences_multi, targets_dx_multi, targets_dy_multi, player_masks_multi, seq_meta_multi, rel_features_multi = _convert_to_multi_player_format(
            sequences, 
            targets_dx, 
            targets_dy, 
            seq_meta,
            Config.WINDOW_SIZE,
            feature_cols,
            df_no_predicted=df_no_predicted,  # ⭐ 传递完整数据用于补充非预测球员
        )
        
        print(f"Multi-player sequences: {len(sequences_multi)} plays, each with 22 players")
        if len(sequences_multi) > 0:
            print(f"  Original features shape: (22, {Config.WINDOW_SIZE}, {len(feature_cols)})")
            print(f"  Relative features shape: (22, {Config.WINDOW_SIZE}, 44)")
            if targets_dx_multi:
                print(f"  Target shapes: (22, {targets_dx_multi[0].shape[1]})")

        if Config.TRAIN:
            return (
                sequences_multi,
                targets_dx_multi,
                targets_dy_multi,
                targets_fids,
                seq_meta_multi,
                player_masks_multi,
                feature_cols,
                rel_features_multi,  # ⭐ 新增
            )
        return sequences_multi, seq_meta_multi, feature_cols, player_masks_multi, rel_features_multi  # ⭐ 新增
    else:
        if Config.TRAIN:
            return (
                sequences,
                targets_dx,
                targets_dy,
                targets_fids,
                seq_meta,
                feature_cols,
            )
        return sequences, seq_meta, feature_cols
