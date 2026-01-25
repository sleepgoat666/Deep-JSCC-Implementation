#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example script demonstrating SWIPT module usage.
This shows how to use the SWIPT functionality for energy-aware communication.
"""

import torch
from torch.utils.data import TensorDataset, DataLoader
from models import DeepJSCC
from metrics import psnr

def example_basic_swipt():
    """Example 1: Basic SWIPT usage."""
    print("=" * 70)
    print("Example 1: Basic SWIPT Usage")
    print("=" * 70)
    
    # Create a model
    model = DeepJSCC(latent_ch=8)
    
    # Create a sample image (batch of 1)
    x = torch.rand(1, 3, 32, 32)
    
    # Standard transmission (no energy harvesting)
    print("\nStandard transmission (no SWIPT):")
    xhat = model(x, snr_db=10, use_swipt=False)
    mse = torch.nn.functional.mse_loss(xhat, x)
    print(f"  Reconstructed image shape: {xhat.shape}")
    print(f"  MSE: {mse.item():.6f}")
    
    # SWIPT transmission with 30% power for energy harvesting
    print("\nSWIPT transmission (ρ=0.3, η=0.8):")
    xhat_swipt, energy = model(x, snr_db=10, use_swipt=True, rho=0.3, eta=0.8)
    mse_swipt = torch.nn.functional.mse_loss(xhat_swipt, x)
    print(f"  Reconstructed image shape: {xhat_swipt.shape}")
    print(f"  MSE: {mse_swipt.item():.6f}")
    print(f"  Harvested energy: {energy.item():.6f}")
    print()

def example_power_splitting_sweep():
    """Example 2: Sweep through different power splitting ratios."""
    print("=" * 70)
    print("Example 2: Power Splitting Ratio Sweep")
    print("=" * 70)
    
    model = DeepJSCC(latent_ch=8)
    x = torch.rand(4, 3, 32, 32)  # Batch of 4 images
    
    print("\nExploring R-E tradeoff by varying ρ:")
    print(f"{'ρ':>6s} | {'MSE':>10s} | {'Energy':>10s} | {'Info Power':>12s}")
    print("-" * 50)
    
    for rho in [0.0, 0.2, 0.4, 0.6, 0.8]:
        xhat, energy = model(x, snr_db=10, use_swipt=True, rho=rho, eta=0.8)
        mse = torch.nn.functional.mse_loss(xhat, x)
        info_power = 1 - rho  # Fraction of power for information
        
        print(f"{rho:6.1f} | {mse.item():10.6f} | {energy.mean().item():10.6f} | {info_power:12.1f}")
    
    print("\nObservation: As ρ increases, more energy is harvested but")
    print("             less power is available for information decoding.\n")

def example_swipt_training():
    """Example 3: Training with SWIPT objective."""
    print("=" * 70)
    print("Example 3: Training with SWIPT Objective")
    print("=" * 70)
    
    model = DeepJSCC(latent_ch=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create synthetic training data
    x_train = torch.rand(100, 3, 32, 32)
    y_train = torch.zeros(100)
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    # SWIPT parameters
    rho = 0.3
    eta = 0.8
    lambda_energy = 0.01
    
    print(f"\nTraining with SWIPT objective:")
    print(f"  ρ = {rho} (power splitting ratio)")
    print(f"  η = {eta} (energy harvesting efficiency)")
    print(f"  λ = {lambda_energy} (energy weight in loss)")
    print(f"\n  Loss = MSE - λ × Energy\n")
    
    # Train for one epoch
    model.train()
    total_loss = 0
    total_mse = 0
    total_energy = 0
    
    for batch_idx, (x, _) in enumerate(loader):
        # Forward pass with SWIPT
        xhat, energy = model(x, snr_db=10, use_swipt=True, rho=rho, eta=eta)
        
        # SWIPT loss: balance reconstruction and energy harvesting
        mse = torch.nn.functional.mse_loss(xhat, x)
        loss = mse - lambda_energy * energy.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_mse += mse.item()
        total_energy += energy.mean().item()
        
        if batch_idx == 0:
            print(f"  First batch:")
            print(f"    MSE: {mse.item():.6f}")
            print(f"    Energy: {energy.mean().item():.6f}")
            print(f"    Loss: {loss.item():.6f}")
    
    n_batches = len(loader)
    print(f"\n  Average after 1 epoch:")
    print(f"    Loss: {total_loss/n_batches:.6f}")
    print(f"    MSE: {total_mse/n_batches:.6f}")
    print(f"    Energy: {total_energy/n_batches:.6f}\n")

def example_re_tradeoff_analysis():
    """Example 4: Analyzing Rate-Energy tradeoff."""
    print("=" * 70)
    print("Example 4: Rate-Energy Tradeoff Analysis")
    print("=" * 70)
    
    model = DeepJSCC(latent_ch=8)
    model.eval()
    
    # Test data
    x_test = torch.rand(50, 3, 32, 32)
    y_test = torch.zeros(50)
    dataset = TensorDataset(x_test, y_test)
    loader = DataLoader(dataset, batch_size=10, shuffle=False)
    
    print("\nAnalyzing R-E tradeoff at different SNR levels:")
    print(f"{'SNR (dB)':>10s} | {'ρ':>6s} | {'PSNR (dB)':>12s} | {'Energy':>10s}")
    print("-" * 50)
    
    with torch.no_grad():
        for snr in [5, 10, 15]:
            for rho in [0.0, 0.3, 0.6]:
                total_psnr = 0
                total_energy = 0
                
                for x, _ in loader:
                    xhat, energy = model(x, snr_db=snr, use_swipt=True, rho=rho, eta=0.8)
                    batch_psnr = psnr(x, xhat)
                    
                    total_psnr += batch_psnr.item() * x.size(0)
                    total_energy += energy.mean().item() * x.size(0)
                
                avg_psnr = total_psnr / len(dataset)
                avg_energy = total_energy / len(dataset)
                
                print(f"{snr:10d} | {rho:6.1f} | {avg_psnr:12.2f} | {avg_energy:10.6f}")
    
    print("\nObservation: Higher SNR generally allows better PSNR.")
    print("             Higher ρ trades off PSNR for more harvested energy.\n")

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SWIPT Module Examples")
    print("Demonstrating Simultaneous Wireless Information and Power Transfer")
    print("=" * 70 + "\n")
    
    example_basic_swipt()
    example_power_splitting_sweep()
    example_swipt_training()
    example_re_tradeoff_analysis()
    
    print("=" * 70)
    print("Examples completed!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Run 'python train_swipt.py' to train a model with SWIPT")
    print("  - Run 'python eval_re_curve.py' to generate R-E curves")
    print("=" * 70 + "\n")
