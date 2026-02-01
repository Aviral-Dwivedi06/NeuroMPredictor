import torch
import pandas as pd
import os
import sys

# Fixed Import: Since we are in the same folder as model.py
from model import CircuitPredictor 

def train_model():
    # Smart Paths: These look one level up from the 'src' folder
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'hardware_specs.csv')
    model_dir = os.path.join(base_dir, 'models')
    save_path = os.path.join(model_dir, 'trained_circuit_model.pth')

    # 1. Load Data
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        sys.exit()

    df = pd.read_csv(data_path)
    
    inputs = torch.tensor(df[['bias', 'cap', 'v_th', 'var']].values).float()
    targets = torch.tensor(df[['freq', 'energy', 'latency']].values).float()

    # 2. Data Normalization
    input_mean, input_std = inputs.mean(dim=0), inputs.std(dim=0)
    target_mean, target_std = targets.mean(dim=0), targets.std(dim=0)

    norm_inputs = (inputs - input_mean) / (input_std + 1e-6)
    norm_targets = (targets - target_mean) / (target_std + 1e-6)

    # 3. Initialize Model
    model = CircuitPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)
    mse = torch.nn.MSELoss()

    print("\n" + "="*60)
    print("STARTING NEUROMORPHIC CIRCUIT CALIBRATION")
    print("="*60)

    # 4. Training Loop
    for epoch in range(1, 1001):
        model.train()
        optimizer.zero_grad()
        predictions = model(norm_inputs)
        
        loss_data = mse(predictions, norm_targets)
        loss_physics = model.physics_loss(norm_inputs, predictions)
        
        p_weight = 0.1 if epoch < 400 else 0.4
        total_loss = loss_data + (p_weight * loss_physics)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(total_loss)

        if epoch % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d}/1000 | Loss: {total_loss.item():.6f} | LR: {current_lr:.5f}")

    # 5. Save Model
    os.makedirs(model_dir, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_mean': input_mean,
        'input_std': input_std,
        'target_mean': target_mean,
        'target_std': target_std
    }
    
    torch.save(checkpoint, save_path)
    print(f"\nSUCCESS: Model saved to {save_path}")

if __name__ == "__main__":
    train_model()