import torch
import torch.nn as nn

class CircuitPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 3) # Outputs: freq, energy, latency
        )

    def forward(self, x):
        return self.net(x)

    def physics_loss(self, inputs, outputs):
        # Extract inputs for LIF equation
        I = inputs[:, 0]
        C = inputs[:, 1]
        Vth = inputs[:, 2]
        pred_freq = outputs[:, 0]
        
        # Constraint: freq - (I / (C * Vth)) = 0
        target_f = I / (C * Vth + 1e-6) * 0.1
        return torch.mean((pred_freq - target_f)**2)