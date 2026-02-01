import torch

def run_prediction(model, input_json, stats):
    model.eval()
    
    # 1. Prepare and Normalize Input Tensor
    # We must use the same mean/std used during training
    feat = torch.tensor([
        input_json["bias_current_nA"], 
        input_json["capacitance_pF"], 
        input_json["threshold_voltage_V"], 
        input_json["rram_variability_percent"]
    ]).float()
    
    # Normalize: (Value - Mean) / Std
    norm_input = (feat - stats['input_mean']) / (stats['input_std'] + 1e-6)
    input_tensor = norm_input.unsqueeze(0)
    input_tensor.requires_grad = True # Enable gradient tracking for sensitivity
    
    # 2. Forward Pass (Prediction in normalized space)
    norm_output = model(input_tensor)
    
    # 3. Denormalize Output to physical units
    # Real Value = (Normalized Value * Std) + Mean
    real_output = (norm_output * stats['target_std']) + stats['target_mean']
    
    # 4. Sensitivity Analysis (Gradients)
    # We calculate gradient of Latency (Index 2) w.r.t Bias Current (Index 0)
    norm_output[0, 2].backward()
    grad_norm = input_tensor.grad[0, 0].item()
    
    # Adjustment for scaling: since we are in normalized space, 
    # the gradient needs to be scaled back to reflect real-world units
    real_grad = grad_norm * (stats['target_std'][2] / stats['input_std'][0])
    
    # Calculate requested sensitivity metrics
    bias_val = input_json["bias_current_nA"]
    bias_drop = bias_val * 0.3
    latency_increase = abs(real_grad * bias_drop)
    
    # Ensure current_latency is a float
    current_latency = float(real_output[0, 2].item())
    
    # Calculate the increase as a float
    # We use float() here to ensure the division doesn't result in a Tensor
    latency_percentage = (float(latency_increase) / current_latency) * 100

    return {
        "spike_frequency_kHz": round(float(real_output[0, 0].item()), 2),
        "energy_per_spike_pJ": round(float(real_output[0, 1].item()), 2),
        "latency_us": round(current_latency, 2),
        "sensitivity": {
            "bias_current": {
                "-30%": {
                    "voltage_drop_percent": 30.0,
                    "latency_increase_percent": round(latency_percentage, 2)
                }
            }
        }
    }