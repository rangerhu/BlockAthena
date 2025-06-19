import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualMLP(nn.Module):
    """Residual MLP with skip connections."""
    def __init__(self, input_dim, hidden_dim, depth):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x  # Residual connection
        return x

def gated_fusion(ddgm, chgm):
    """Intra-period gated fusion of DDGM and CHGM features."""
    fused = []
    w = torch.nn.Parameter(torch.randn(ddgm.shape[1] + chgm.shape[1]))
    W_f = torch.nn.Linear(ddgm.shape[1] + chgm.shape[1], ddgm.shape[1])
    b_f = torch.nn.Parameter(torch.zeros(ddgm.shape[1]))

    for d, h in zip(ddgm, chgm):
        x = np.concatenate([d, h])
        x_tensor = torch.tensor(x, dtype=torch.float32)
        weight = torch.exp(torch.tanh(x_tensor) @ w)
        fused_vec = F.relu(W_f(x_tensor) + b_f) * weight
        fused.append(fused_vec.detach().numpy())
    return np.stack(fused)

def weighted_aggregation(fused, energies):
    """Inter-period spectral-energy weighted aggregation with residual modeling."""
    beta = energies / np.sum(energies)
    z0 = np.sum([b * f for b, f in zip(beta, fused)], axis=0)
    model = ResidualMLP(input_dim=z0.shape[0], hidden_dim=64, depth=3)
    z = torch.tensor(z0, dtype=torch.float32).unsqueeze(0)
    z_final = model(z)
    return z_final.detach().numpy().squeeze()

def detect_eth_crime(z_final, y_binary, y_multi):
    """Simple two-stage classifier for Ethereum crime detection."""
    # Stage 1: SVM for binary classification
    from sklearn.svm import SVC
    clf_bin = SVC()
    clf_bin.fit([z_final], [y_binary])  # Placeholder fit
    result = clf_bin.predict([z_final])[0]

    # Stage 2: FCNN for multi-class classification if crime detected
    if result == 1:
        class FCNN(nn.Module):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Linear(in_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, out_dim)
                )
            def forward(self, x): return self.model(x)

        model = FCNN(len(z_final), 4)  # Assume 4 crime categories
        output = model(torch.tensor(z_final, dtype=torch.float32))
        pred_class = torch.argmax(output).item()
        return True, pred_class
    else:
        return False, -1
