import pandas as pd
import numpy as np
import os

# Ensure the directory exists
os.makedirs('data', exist_ok=True)

def generate_hardware_data(samples=1000):
    np.random.seed(42)
    
    # Inputs: Hardware Parameters
    bias = np.random.uniform(50, 150, samples)          # nA
    cap = np.random.uniform(0.5, 2.5, samples)           # pF
    v_th = np.random.uniform(0.3, 0.6, samples)          # V
    var = np.random.uniform(5, 20, samples)             # RRAM Variability %
    
    # Target Outputs: Based on Neuromorphic Physics
    # Spike Frequency (kHz) ~ I / (C * Vth)
    freq = (bias / (cap * v_th)) * 0.1 
    
    # Energy (pJ) - Based on 15.2 pJ benchmark from LSMCore/JETC docs
    # Energy increases with capacitance and variability
    energy = 15.2 + (cap * 2.0) + (var * 0.15)
    
    # Latency (us) - Inversely proportional to frequency
    latency = 30 + (15 / (freq + 0.1))

    df = pd.DataFrame({
        'bias': bias,
        'cap': cap,
        'v_th': v_th,
        'var': var,
        'freq': freq,
        'energy': energy,
        'latency': latency
    })
    
    # Save with specific formatting to avoid ParserError
    df.to_csv('data/hardware_specs.csv', index=False)
    print("Success: Generated 1000 clean samples in data/hardware_specs.csv")

if __name__ == "__main__":
    generate_hardware_data()