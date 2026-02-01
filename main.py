import torch
import json
import os
import sys
from src.model import CircuitPredictor
from src.sensitivity import run_prediction

# --- PHYSICAL SAFETY LIMITS ---
LIMITS = {
    "VTH_MAX": 1.5,    # Volts: Above this, oxide breakdown occurs
    "VTH_MIN": 0.1,    # Volts: Below this, noise triggers random firing
    "BIAS_MAX": 500.0, # nA: High current leads to thermal runaway
    "BIAS_MIN": 1.0,   # nA: Below this, signal is lost in thermal noise
    "VAR_MAX": 40.0,   # %: Beyond 40%, the RRAM crossbar loses state reliability
    "VAR_MIN": 0,      # %: Ideal theoretical device with zero fluctuations
}

def check_safety(params):
    warnings = []
    if params["threshold_voltage_V"] > LIMITS["VTH_MAX"]:
        warnings.append(f"DANGER: Vth ({params['threshold_voltage_V']}V) exceeds Gate Oxide breakdown limits.")
    if params["threshold_voltage_V"] < LIMITS["VTH_MIN"]:
        warnings.append("STABILITY WARNING: Vth is too low. Expect random 'Ghost Spiking'.")
    if params["bias_current_nA"] > LIMITS["BIAS_MAX"]:
        warnings.append(f"THERMAL WARNING: Bias ({params['bias_current_nA']}nA) may cause device overheating.")
    if params["rram_variability_percent"] > LIMITS["VAR_MAX"]:
        warnings.append("ERROR: RRAM Variability too high. Memristive state is non-functional.")
    if params["rram_variability_percent"] < LIMITS["VAR_MIN"]:
        warnings.append("ERROR! PHYSICAL IMPOSSIBILITY: RRAM Variability cannot be negative.")
    if params["bias_current_nA"] < LIMITS["BIAS_MIN"]:
        warnings.append(f"SIGNAL INTEGRITY WARNING: Bias ({params['bias_current_nA']}nA) is below the noise floor; spikes may be masked by thermal fluctuations.")
    return warnings

def interactive_session(model, stats):
    # Baseline Specs from LSMCore/JETC Research
    state = {
        "bias_current_nA": 100.0,
        "capacitance_pF": 1.5,
        "threshold_voltage_V": 0.4,
        "rram_variability_percent": 10.0
    }

    while True:
        print("\n" + "-"*55)
        print("CURRENT HARDWARE STATE (SIMULATED):")
        print(f"  Bias: {state['bias_current_nA']:.1f}nA | Cap: {state['capacitance_pF']:.1f}pF")
        print(f"  Vth:  {state['threshold_voltage_V']:.2f}V   | RRAM Var: {state['rram_variability_percent']:.1f}%")
        print("-" * 55)
        print("WHAT-IF SCENARIOS (Quantitative Control):")
        print("1. Adjust Bias Current (nA)      [Simulate Power Throttling]")
        print("2. Adjust Threshold Voltage (V)  [Simulate Sensitivity Tuning]")
        print("3. Adjust RRAM Variability (%)   [Simulate Device Aging/Noise]")
        print("4. Reset to Research Baseline    [LSMCore 400MHz Specs]")
        print("5. Exit Simulator")
        
        choice = input("\nSelect an action (1-5): ")

        try:
            if choice == '1':
                current = state["bias_current_nA"]
                max_up = LIMITS["BIAS_MAX"] - current
                max_down = LIMITS["BIAS_MIN"] - current
                print(f"\n>>> Current Bias: {current:.1f} nA")
                print(f">>> Safe Change Range: {max_down:+.1f} nA to {max_up:+.1f} nA")
                delta = float(input("Enter change in nA: "))
                state["bias_current_nA"] += delta

            elif choice == '2':
                current = state["threshold_voltage_V"]
                max_up = LIMITS["VTH_MAX"] - current
                max_down = LIMITS["VTH_MIN"] - current
                print(f"\n>>> Current Vth: {current:.2f} V")
                print(f">>> Safe Change Range: {max_down:+.2f} V to {max_up:+.2f} V")
                delta = float(input("Enter change in Voltage (V): "))
                state["threshold_voltage_V"] += delta

            elif choice == '3':
                current = state["rram_variability_percent"]
                max_up = LIMITS["VAR_MAX"] - current
                max_down = LIMITS["VAR_MIN"] - current
                print(f"\n>>> Current Variability: {current:.1f}%")
                print(f">>> Safe Change Range: {max_down:+.1f}% to {max_up:+.1f}%")
                delta = float(input("Enter change in %: "))
                state["rram_variability_percent"] += delta

            elif choice == '4':
                state = {"bias_current_nA": 100.0, "capacitance_pF": 1.5, "threshold_voltage_V": 0.4, "rram_variability_percent": 10.0}
                print(">>> State Reset to Baseline.")
                continue

            elif choice == '5':
                print("Simulator Shutdown.")
                break
            else: continue

        except ValueError:
            print("Invalid input. Please enter numerical values.")
            continue

        warnings = check_safety(state)
        if warnings:
            for w in warnings: print(w)
            print("! THEORETICAL PREDICTION STILL CALCULATED BELOW !")

        # Prediction now uses the saved stats for proper scaling
        result = run_prediction(model, state, stats)
        total_power_nw = result["spike_frequency_kHz"] * result["energy_per_spike_pJ"]
        
        print("\n>>> PREDICTED IMPACT:")
        print(json.dumps(result, indent=4))
        print(f"\nESTIMATED POWER CONSUMPTION: {total_power_nw:.2f} nW")

if __name__ == "__main__":
    model_path = 'models/trained_circuit_model.pth'
    
    if not os.path.exists(model_path):
        print("Error: Trained model not found. Please run 'python train.py' first.")
        sys.exit()

    # Load the high-precision checkpoint
    checkpoint = torch.load(model_path)
    
    my_model = CircuitPredictor()
    my_model.load_state_dict(checkpoint['model_state_dict'])
    my_model.eval()

    # Extract normalization constants needed for prediction
    stats = {
        'input_mean': checkpoint['input_mean'],
        'input_std': checkpoint['input_std'],
        'target_mean': checkpoint['target_mean'],
        'target_std': checkpoint['target_std']
    }

    print("\n>>> Hardware-Aware Model Loaded Successfully.")
    interactive_session(my_model, stats)