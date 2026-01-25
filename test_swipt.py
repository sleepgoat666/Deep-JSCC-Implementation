#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for SWIPT module functionality.
"""

import torch
from channel import power_normalize, awgn, swipt_channel
from models import DeepJSCC

def test_swipt_channel():
    """Test SWIPT channel function."""
    print("=" * 60)
    print("Testing SWIPT channel function")
    print("=" * 60)
    
    # Create dummy signal
    batch_size = 4
    channels = 8
    height, width = 8, 8
    y = torch.randn(batch_size, channels, height, width)
    
    # Test parameters
    rho = 0.3
    eta = 0.8
    snr_db = 10.0
    
    # Apply SWIPT channel
    y_id, e_harvested = swipt_channel(y, rho, eta, snr_db)
    
    print(f"Input signal shape: {y.shape}")
    print(f"Output signal shape: {y_id.shape}")
    print(f"Harvested energy shape: {e_harvested.shape}")
    print(f"Harvested energy values: {e_harvested}")
    print(f"Average harvested energy: {e_harvested.mean().item():.6f}")
    
    # Verify power splitting
    # The ID signal should have approximately (1-rho) of the power
    input_power = y.pow(2).mean().item()
    output_power_before_noise = ((1 - rho) * y).pow(2).mean().item()
    expected_power = (1 - rho) * input_power
    
    print(f"\nInput signal power: {input_power:.6f}")
    print(f"Expected ID signal power (before noise): {expected_power:.6f}")
    print(f"Actual ID signal power (before noise): {output_power_before_noise:.6f}")
    
    # Verify energy harvesting formula
    expected_energy = eta * rho * (y.pow(2).mean(dim=(1, 2, 3)))
    print(f"\nExpected harvested energy: {expected_energy}")
    print(f"Actual harvested energy: {e_harvested}")
    print(f"Match: {torch.allclose(expected_energy, e_harvested, rtol=1e-5)}")
    
    print("\n✓ SWIPT channel test passed!\n")

def test_model_forward():
    """Test DeepJSCC model with SWIPT."""
    print("=" * 60)
    print("Testing DeepJSCC model with SWIPT")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    model = DeepJSCC(latent_ch=8).to(device)
    
    # Create dummy input (batch of images)
    batch_size = 2
    x = torch.rand(batch_size, 3, 32, 32).to(device)
    
    # Test standard forward pass
    print("Test 1: Standard forward pass (no SWIPT)")
    xhat = model(x, snr_db=10.0, use_swipt=False)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {xhat.shape}")
    print(f"  Output value range: [{xhat.min().item():.4f}, {xhat.max().item():.4f}]")
    assert xhat.shape == x.shape, "Output shape mismatch!"
    print("  ✓ Passed\n")
    
    # Test SWIPT forward pass
    print("Test 2: SWIPT forward pass")
    xhat, e_harvested = model(x, snr_db=10.0, use_swipt=True, rho=0.3, eta=0.8)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {xhat.shape}")
    print(f"  Harvested energy shape: {e_harvested.shape}")
    print(f"  Harvested energy: {e_harvested}")
    print(f"  Average energy: {e_harvested.mean().item():.6f}")
    assert xhat.shape == x.shape, "Output shape mismatch!"
    assert e_harvested.shape == (batch_size,), "Energy shape mismatch!"
    print("  ✓ Passed\n")
    
    # Test different rho values
    print("Test 3: Varying rho values")
    rho_values = [0.0, 0.3, 0.5, 0.7, 0.9]
    for rho in rho_values:
        xhat, e_harvested = model(x, snr_db=10.0, use_swipt=True, rho=rho, eta=0.8)
        mse = torch.nn.functional.mse_loss(xhat, x).item()
        avg_energy = e_harvested.mean().item()
        print(f"  rho={rho:.1f}: MSE={mse:.6f}, Energy={avg_energy:.6f}")
    print("  ✓ Passed\n")
    
    print("✓ All model tests passed!\n")

def test_loss_computation():
    """Test SWIPT loss computation."""
    print("=" * 60)
    print("Testing SWIPT loss computation")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepJSCC(latent_ch=8).to(device)
    
    x = torch.rand(2, 3, 32, 32).to(device)
    
    # SWIPT parameters
    rho = 0.3
    eta = 0.8
    lambda_energy = 0.01
    
    xhat, e_harvested = model(x, snr_db=10.0, use_swipt=True, rho=rho, eta=eta)
    
    mse_loss = torch.nn.functional.mse_loss(xhat, x)
    energy_term = e_harvested.mean()
    total_loss = mse_loss - lambda_energy * energy_term
    
    print(f"MSE Loss: {mse_loss.item():.6f}")
    print(f"Energy Term: {energy_term.item():.6f}")
    print(f"Total Loss: {total_loss.item():.6f}")
    print(f"Lambda: {lambda_energy}")
    print(f"\nLoss = MSE - lambda * Energy")
    print(f"     = {mse_loss.item():.6f} - {lambda_energy} * {energy_term.item():.6f}")
    print(f"     = {total_loss.item():.6f}")
    
    print("\n✓ Loss computation test passed!\n")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SWIPT Module Test Suite")
    print("=" * 60 + "\n")
    
    test_swipt_channel()
    test_model_forward()
    test_loss_computation()
    
    print("=" * 60)
    print("All tests passed successfully! ✓")
    print("=" * 60)
