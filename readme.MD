# (NeuroM Predictor)
## Neuromorphic Framework Predictor

This project was designed for the Micron Mimory Competition by Team Morpheus. 
By combining machine learning with semiconductor physics, we aim to provide a tool that reduces the need for expensive SPICE simulations while maintaining the accuracy required for next-generation RRAM neuromorphic chip design.

## Project Overview

This repository contains a Physics-Informed Neural Network (PINN) designed to simulate and predict the performance metrics of a Leaky Integrate-and-Fire (LIF) neuron circuit. The architecture is specifically modeled after the LSMCore (40nm) and RRAM-based crossbar arrays.

The system leverages deep learning to bridge the gap between abstract neural simulations and physical hardware constraints.

---

## Research & Literature Context

This model is grounded in current neuromorphic research regarding the trade-offs between stochasticity, power, and reliability in non-volatile memory (NVM) systems.

* **LSMCore Architecture:** The simulator logic follows the 40nm CMOS implementation of Liquid State Machines, focusing on the energy-delay product of spiking neurons.
* **Memristive Stochasticity:** Literature suggests that RRAM variability follows a log-normal distribution. Our model incorporates this as a "Reliability Factor," where variability  leads to synaptic weight decoherence.
* **Thermal Noise Floor:** Based on Johnson-Nyquist noise theory, we implement a minimum bias current limit to ensure that the stochastic nature of electron flow at the nano-scale does not override the intentional spiking signal.

---

## Mathematical Framework

The model is constrained by the following physical laws of VLSI circuit design.

### 1. The LIF Charging Equation

The fundamental relationship between bias current () and the time-to-spike () is governed by:

$$I_{bias} = C \cdot \frac{V_{th} - V_{reset}}{t_{lat}} + I_{leak}$$

Where:

* $C$ : Membrane capacitance.
* $V_{th}$ : Threshold voltage.
* $I_{leak}$ : Leakage current (exponentially dependent on ).

### 2. Spike Frequency ()

Predicting the frequency requires accounting for the refractory period ():

$$f = \frac{1}{t_{lat} + t_{ref}}$$

### 3. Physics-Informed Loss Function

During training, the model minimizes a composite loss function ():

$$L_{total} = \lambda_{data} \cdot MSE(y_{pred}, y_{true}) + \lambda_{physics} \cdot \| \frac{I_{bias} \cdot t_{lat}}{C} - V_{th} \|^2$$

This ensures that as the model learns from the Micron Mimory datasets, it remains tethered to the conservation of charge.

---

## Hardware Specifications and Limits

| Parameter | Minimum | Maximum | Hardware Risk |
| --- | --- | --- | --- |
| Bias Current | 1.0 nA | 500.0 nA | Signal Integrity / Thermal Runaway |
| Threshold Voltage | 0.1 V | 1.5 V | Ghost Spiking / Oxide Breakdown |
| RRAM Variability | 0.0 % | 40.0 % | Memristive State Reliability |

---

## Technical Architecture

### Core Model (`src/model.py`)

A deep feed-forward network with:

* **Input Scaling:** Z-score normalization to handle the magnitude difference between nA and Volts.
* **LeakyReLU Activation:** Prevents dead neurons during the backpropagation of complex physics gradients.
* **Batch Normalization:** Stabilizes training across diverse hardware configurations.

### Sensitivity Engine (`src/sensitivity.py`)

Calculates the first-order derivative of the hardware's response to power fluctuations. The sensitivity  $S$  is defined as:

$$S = \frac{\partial \text{Latency}}{\partial I_{bias}}$$

This allows the simulator to predict how a 30% drop in supply current will degrade the timing accuracy of the neural network.

---

## Installation and Usage

### 1. Calibration (Training)

To train the model using the high-precision adaptive weighting:

```bash
python src/train.py

```

### 2. Simulation (Inference)

Launch the interactive "What-If" simulation dashboard:

```bash
python main.py

```
---

