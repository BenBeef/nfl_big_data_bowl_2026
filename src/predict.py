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
