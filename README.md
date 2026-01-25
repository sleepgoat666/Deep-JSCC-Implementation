# Deep JSCC implementation 

+ Minimal PyTorch implementation of Deep Joint Source-Channel Coding (DeepJSCC) over an AWGN channel.
+ Trains separate models at fixed SNR_train ∈ {1,4,7,13,19} and evaluates PSNR across SNR_test sweep.
+ This implementation is based on the paper [Deep Joint Source-Channel Coding for Wireless Image Transmission](https://ieeexplore.ieee.org/abstract/document/8723589).
+ **NEW**: Includes SWIPT (Simultaneous Wireless Information and Power Transfer) module for energy harvesting.

+ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/drive/16rcHuqdTN3X9q8ojp25UNF-BZUXzhOqT
)

## Architecture
![Architecture](architecture.jpg)

## Standard Deep JSCC Training

### Train
python train.py 

### Evaluation
python eval.py

### Result
![AWGN PSNR](awgn_psnr.png)

## SWIPT Module

The SWIPT (Simultaneous Wireless Information and Power Transfer) module enables energy harvesting alongside information transmission. The receiver splits the received signal power into two parts:
- **(1-ρ)** fraction for information decoding (ID)
- **ρ** fraction for energy harvesting (EH)

### SWIPT Parameters

- **ρ (rho)**: Power splitting ratio for energy harvesting (0 ≤ ρ ≤ 1)
  - ρ = 0: All power for information decoding (no energy harvesting)
  - ρ = 1: All power for energy harvesting (no information decoding)
  - Typical values: 0.1 to 0.5

- **η (eta)**: Energy harvesting efficiency (0 ≤ η ≤ 1)
  - Represents the efficiency of converting RF power to DC power
  - Typical values: 0.5 to 0.9
  - Default: 0.8 (80% efficiency)

- **λ (lambda_energy)**: Weight for energy optimization in loss function
  - Controls the trade-off between reconstruction quality and energy harvesting
  - Loss = MSE - λ × E_harvested
  - Typical values: 0.001 to 0.1
  - Default: 0.01

### Energy Harvesting Computation

The harvested energy is computed as:
```
E_harvested = η · ρ · |y|²
```
where `y` is the received signal.

### Quick Start: SWIPT Examples

Run the example script to see SWIPT in action:
```bash
python example_swipt.py
```

This demonstrates:
- Basic SWIPT usage
- Power splitting ratio sweep
- Training with SWIPT objective
- Rate-Energy tradeoff analysis

### SWIPT Training

Train Deep JSCC with SWIPT using:
```bash
python train_swipt.py
```

This trains models with:
- Power splitting ratio ρ = 0.3 (30% for energy harvesting)
- Energy harvesting efficiency η = 0.8 (80%)
- Energy weight λ = 0.01
- Training SNRs: {7, 13, 19} dB

The trained models are saved in `checkpoints_swipt/` directory.

### R-E Curve Evaluation

Evaluate the Rate-Energy (R-E) tradeoff by running:
```bash
python eval_re_curve.py
```

This script:
1. Varies ρ from 0.0 to 0.9
2. Measures PSNR and harvested energy for each ρ value
3. Generates three plots:
   - PSNR vs ρ: Shows how reconstruction quality degrades as more power is allocated to energy harvesting
   - Harvested Energy vs ρ: Shows energy harvesting increases with ρ
   - Rate-Energy Tradeoff: Shows the fundamental tradeoff between information rate (PSNR) and harvested energy

Results are saved in `results/` directory.

### Using SWIPT in Code

```python
from models import DeepJSCC

model = DeepJSCC(latent_ch=8)

# Forward pass with SWIPT
xhat, e_harvested = model(x, snr_db=10, use_swipt=True, rho=0.3, eta=0.8)

# Standard forward pass (no SWIPT)
xhat = model(x, snr_db=10, use_swipt=False)
```

### Dynamic Power Splitting

You can dynamically adjust ρ during training or evaluation to explore different operating points on the R-E curve. The `eval_re_curve.py` script demonstrates this by sweeping through different ρ values.

### Running Tests

Verify the SWIPT implementation with the test suite:
```bash
python test_swipt.py              # Test SWIPT channel and model
python test_swipt_training.py     # Test training loop
python test_re_eval.py            # Test R-E evaluation
```

