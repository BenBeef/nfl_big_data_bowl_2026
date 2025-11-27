import torch
import numpy as np


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


def predict_sst_separate_1d(model_1d, scaler, X_test_raw, device):
    """
    使用单维度模型进行预测
    
    Args:
        model_1d: 单维度模型（STTransformer1D）
        scaler: 数据标准化器
        X_test_raw: 原始测试数据列表
        device: 设备
        
    Returns:
        (B, H) 形状的预测数组
    """
    model_1d.eval()
    base = np.stack([scaler.transform(s) for s in X_test_raw]).astype(np.float32)
    xt = torch.tensor(base, device=device)

    with torch.no_grad():
        output = model_1d(xt)  # (B, H)
    
    return output.detach().cpu().numpy()


def predict_sst_separate_models(model_x, model_y, scaler, X_test_raw, device):
    """
    使用分离的 X 和 Y 模型进行预测
    
    Args:
        model_x: X 维度模型（STTransformer1D）
        model_y: Y 维度模型（STTransformer1D）
        scaler: 数据标准化器
        X_test_raw: 原始测试数据列表
        device: 设备
        
    Returns:
        (dx, dy) 预测数组，每个形状为 (B, H)
    """
    dx = predict_sst_separate_1d(model_x, scaler, X_test_raw, device)
    dy = predict_sst_separate_1d(model_y, scaler, X_test_raw, device)
    
    return dx, dy
