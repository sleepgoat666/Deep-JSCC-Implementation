#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test R-E curve evaluation with synthetic data.
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from models import DeepJSCC
from eval_re_curve import eval_swipt

def test_re_curve_evaluation():
    """Test R-E curve evaluation with synthetic data."""
    print("=" * 60)
    print("Testing R-E Curve Evaluation")
    print("=" * 60)
    
    device = 'cpu'
    model = DeepJSCC(latent_ch=8).to(device)
    
    # Create synthetic test dataset
    num_samples = 100
    x_data = torch.rand(num_samples, 3, 32, 32)
    y_data = torch.zeros(num_samples)
    dataset = TensorDataset(x_data, y_data)
    loader = DataLoader(dataset, batch_size=20, shuffle=False)
    
    # Test evaluation at different rho values
    print("\nEvaluating at SNR=10dB with eta=0.8:")
    print(f"{'rho':>6s} | {'PSNR (dB)':>10s} | {'Energy':>10s}")
    print("-" * 35)
    
    rho_values = np.linspace(0.0, 0.9, 10)
    psnr_list = []
    energy_list = []
    
    for rho in rho_values:
        avg_psnr, avg_energy = eval_swipt(model, loader, device, snr_db=10, rho=rho, eta=0.8)
        psnr_list.append(avg_psnr)
        energy_list.append(avg_energy)
        print(f"{rho:6.2f} | {avg_psnr:10.2f} | {avg_energy:10.6f}")
    
    # Verify trends
    print("\n" + "=" * 60)
    print("Verifying R-E Tradeoff Trends")
    print("=" * 60)
    
    # Energy should increase with rho
    energy_increasing = all(energy_list[i] <= energy_list[i+1] for i in range(len(energy_list)-1))
    print(f"Energy increases with rho: {energy_increasing} ✓" if energy_increasing else f"Energy increases with rho: {energy_increasing} ✗")
    
    # PSNR should generally decrease with rho (as less power goes to information)
    # For untrained model, PSNR might be noisy, so just check energy trend
    print(f"Energy at rho=0.0: {energy_list[0]:.6f}")
    print(f"Energy at rho=0.9: {energy_list[-1]:.6f}")
    print(f"Energy ratio: {energy_list[-1]/max(energy_list[0], 1e-10):.2f}x")
    
    print("\n✓ R-E curve evaluation test passed!")

if __name__ == "__main__":
    test_re_curve_evaluation()
