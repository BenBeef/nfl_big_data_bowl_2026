from typing import * 
import numpy as np 
from sklearn.model_selection import GroupKFold
from src.model import train_all_folds_stt
from src.config import Config
from src.utils import set_seed, load_input_output, write_meta
from src.preprocess import prepare_sequences_with_advanced_features
from datetime import datetime

timestamp = lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S')


top_k = 64
selected_features = ['velocity_y_diff_lag1', 'velocity_y_change', 'velocity_x_change', 'velocity_x_diff_lag1', 'velocity_y_dev3', 'ball_direction_y', 'velocity_y_lag5', 'velocity_y_lag10', 'velocity_x_dev3', 'ball_direction_x', 'velocity_y_dev10', 'velocity_y_diff_lag2', 'velocity_y_roll10', 'time_dist_urgency', 's_change', 's_diff_lag1', 'velocity_x_dev10', 'error_from_ball_y', 'velocity_x_lag5', 'a_change', 'velocity_x_lag10', 'velocity_x_diff_lag2', 'distance_to_ball', 'ball_land_y', 'velocity_x_roll10', 'dist_scaled_by_progress', 'velocity_y_dev20', 'kinetic_energy', 'velocity_perpendicular', 'land_lateral_offset', 'density_ratio', 'velocity_y_roll5', 'velocity_y_lag3', 'speed_squared', 'error_from_ball_x', 's_dev3', 'orientation_diff', 'momentum_y', 's_std5', 'velocity_y', 'dist_from_sideline', 'd2ball_ddt', 'velocity_x_lag3', 'velocity_x_dev20', 'receiver_distance', 's_lag5', 'dvx_opp_w_sum', 'time_to_end', 'velocity_y_lag2', 'y', 'dvy_opp_w_sum', 'x', 's_dev10', 'passer_distance', 'error_from_ball', 'velocity_y_roll20', 'o_change', 's_roll10', 'velocity_y_std3', 'momentum_x', 'velocity_x_roll5', 'v_to_receiver_alignment', 'velocity_y_roll3', 'receiver_deviation', 'time_normalized_all', 'velocity_x', 'velocity_x_std5', 'ball_land_x', 'defender_pressure', 'dx_ally_w_sum', 'velocity_x_roll3', 'velocity_x_roll20', 's_diff_lag2', 'a', 'velocity_y_lag1', 'velocity_y_dev5', 'velocity_x_lag2', 'expected_y_at_ball', 'velocity_x_lag1', 'v_to_passer_alignment', 'time_urgency', 'dist_from_endzone', 'speed_trend_ratio', 's_lag10', 'curvature_abs', 'dvx_ally_w_sum', 'dy_ally_w_sum', 'v_to_passer_perp', 'velocity_x_std10', 'velocity_x_std20', 'velocity_y_std5', 'velocity_y_std10', 'velocity_y_diff_lag3', 'time_normalized_pass', 'bearing_to_receiver', 'time_to_intercept', 'v_to_receiver_perp', 'field_zone_x', 'dx_opp_w_sum', 'speed_mean', 's_diff_lag3', 'accel_alignment', 'curvature_signed', 'traj_width', 'angle_diff', 'velocity_x_std3', 'pressure', 'speed_change', 'field_zone_y', 'o', 'angle_to_ball', 's_roll5', 's', 'd2ball_dt', 'dist_from_center', 'dist_ally_mean', 's_std3', 'velocity_alignment', 'under_pressure', 's_roll20', 'receiver_optimality', 'in_red_zone', 'gnn_n1_dist', 'geo_endpoint_y', 'bearing_to_land_signed', 'acceleration_y', 'frame_id', 'velocity_x_diff_lag3', 'dir', 'dvy_ally_w_sum', 'accel_perpendicular', 'gnn_n3_dist', 's_lag1', 'bearing_to_passer', 'player_height_feet', 'dist_ally_min', 'is_receiver', 'is_offense', 's_lag2', 'traj_direction_angle', 'acceleration_x', 'is_coverage', 's_lag3', 'pressure_speed', 'defender_closing_speed', 'velocity_x_dev5', 's_dev5', 'dist_opp_min', 'closing_speed', 'traj_depth', 's_dev20', 'velocity_y_std20', 'dy_opp_w_sum', 's_std10', 'geo_endpoint_x', 'bmi', 'is_defense', 'gnn_n2_dist', 'player_weight', 'oppn_density', 'have_assistance', 'receiver_speed_usage', 'traj_straightness', 'is_opp_sum', 'dist_opp_mean', 'dir_change', 'is_ally_sum', 's_std20', 'traj_max_turn', 's_roll3', 'traj_mean_turn', 'near_sideline', 'is_passer', 'ally_density']
selected_features = selected_features[:top_k]

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
    result = prepare_sequences_with_advanced_features(train_input, train_output, feature_groups, selected_features)
    sequences, targets_dx, targets_dy, targets_fids, seq_meta, feature_cols = result

    
    print(f"\n[3/4] [{timestamp()}] Training all folds...")
    
    input_dim = len(feature_cols)
    seed = Config.SEEDS[0]
    gkf = GroupKFold(n_splits=Config.N_FOLDS)
    groups = np.array([d['game_id'] for d in seq_meta])

    train_all_folds_stt(gkf, sequences, groups, targets_dx, targets_dy, seed, input_dim)


    print(f"\n[4/4] [{timestamp()}] Save Meta data...")
    write_meta(
        feature_cols = feature_cols, 
        base_dir=Config.SAVE_DIR,
        feature_groups = feature_groups,
        save_src=True
    )

    print(f"\n[{timestamp()}] finished..")