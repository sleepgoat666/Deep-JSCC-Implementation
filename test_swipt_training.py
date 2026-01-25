#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple test for SWIPT training without requiring dataset download.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from models import DeepJSCC
from train_swipt import train_one_epoch_swipt

def test_swipt_training():
    """Test SWIPT training loop with synthetic data."""
    print("=" * 60)
    print("Testing SWIPT Training Loop")
    print("=" * 60)
    
    device = 'cpu'
    model = DeepJSCC(latent_ch=8).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create synthetic dataset (random images)
    num_samples = 100
    x_data = torch.rand(num_samples, 3, 32, 32)
    y_data = torch.zeros(num_samples)  # Dummy labels
    dataset = TensorDataset(x_data, y_data)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    # Test SWIPT training
    print("\nTraining with SWIPT parameters:")
    print("  SNR: 10 dB")
    print("  rho: 0.3 (30% for energy harvesting)")
    print("  eta: 0.8 (80% efficiency)")
    print("  lambda: 0.01")
    
    loss, mse, energy = train_one_epoch_swipt(
        model, loader, opt, device, 
        snr_train=10, rho=0.3, eta=0.8, lambda_energy=0.01
    )
    
    print(f"\nResults after 1 epoch:")
    print(f"  Total Loss: {loss:.6f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  Average Harvested Energy: {energy:.6f}")
    
    # Test with different rho values
    print("\n" + "=" * 60)
    print("Testing different power splitting ratios")
    print("=" * 60)
    
    rho_values = [0.0, 0.2, 0.4, 0.6, 0.8]
    for rho in rho_values:
        model_test = DeepJSCC(latent_ch=8).to(device)
        opt_test = torch.optim.Adam(model_test.parameters(), lr=1e-3)
        
        loss, mse, energy = train_one_epoch_swipt(
            model_test, loader, opt_test, device, 
            snr_train=10, rho=rho, eta=0.8, lambda_energy=0.01
        )
        
        print(f"rho={rho:.1f}: Loss={loss:.6f}, MSE={mse:.6f}, Energy={energy:.6f}")
    
    print("\n✓ SWIPT training test passed!")

if __name__ == "__main__":
    test_swipt_training()
