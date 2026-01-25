# -*- coding: utf-8 -*-
"""
R-E Curve Evaluation for SWIPT-enabled Deep JSCC.
Computes Rate-Energy curves by varying rho values.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import DeepJSCC
from metrics import psnr

@torch.no_grad()
def eval_swipt(model, loader, device, snr_db, rho, eta):
    """
    Evaluate PSNR and harvested energy for given SWIPT parameters.
    
    Args:
        model: DeepJSCC model
        loader: Data loader
        device: Device (cuda/cpu)
        snr_db: SNR in dB
        rho: Power splitting ratio
        eta: Energy harvesting efficiency
    
    Returns:
        avg_psnr, avg_energy
    """
    model.eval()
    total_psnr = 0.0
    total_energy = 0.0
    
    for x, _ in loader:
        x = x.to(device)
        xhat, e_harvested = model(x, snr_db, use_swipt=True, rho=rho, eta=eta)
        
        batch_psnr = psnr(x, xhat).item()
        batch_energy = e_harvested.mean().item()
        
        total_psnr += batch_psnr * x.size(0)
        total_energy += batch_energy * x.size(0)
    
    n = len(loader.dataset)
    return total_psnr / n, total_energy / n

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[device]", device, flush=True)

    # Parameters
    train_snr_list = [7, 13, 19]
    test_snr = 13  # Test at fixed SNR
    latent_ch = 8
    eta = 0.8  # Energy harvesting efficiency
    
    # Vary rho from 0 to 0.9 to compute R-E curve
    rho_values = np.linspace(0.0, 0.9, 10)

    tfm = transforms.Compose([transforms.ToTensor()])
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0,
                             pin_memory=(device == "cuda"))

    os.makedirs("results", exist_ok=True)
    
    # Compute R-E curves for each trained model
    plt.figure(figsize=(12, 5))
    
    # Plot 1: PSNR vs rho
    plt.subplot(1, 2, 1)
    for snr_tr in train_snr_list:
        # Try to load SWIPT-trained model, fall back to regular model
        swipt_ckpt = f"checkpoints_swipt/deepjscc_swipt_snrtrain_{snr_tr}dB_rho0.3.pth"
        regular_ckpt = f"checkpoints/deepjscc_snrtrain_{snr_tr}dB.pth"
        
        if os.path.exists(swipt_ckpt):
            ckpt_path = swipt_ckpt
            label_prefix = "SWIPT"
        elif os.path.exists(regular_ckpt):
            ckpt_path = regular_ckpt
            label_prefix = "Regular"
        else:
            print(f"Warning: No checkpoint found for SNR_train={snr_tr}dB")
            continue
        
        print(f"Loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        
        model = DeepJSCC(latent_ch=latent_ch).to(device)
        model.load_state_dict(ckpt["state_dict"])
        
        psnr_list = []
        energy_list = []
        
        for rho in rho_values:
            avg_psnr, avg_energy = eval_swipt(model, test_loader, device, test_snr, rho, eta)
            psnr_list.append(avg_psnr)
            energy_list.append(avg_energy)
            print(f"  SNR_train={snr_tr}dB, rho={rho:.2f}: PSNR={avg_psnr:.2f}dB, Energy={avg_energy:.6f}")
        
        plt.plot(rho_values, psnr_list, marker='o', label=f"{label_prefix} (SNR_train={snr_tr}dB)")
    
    plt.xlabel('Power Splitting Ratio (ρ)')
    plt.ylabel('PSNR (dB)')
    plt.title(f'PSNR vs ρ @ SNR_test={test_snr}dB')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Energy vs rho
    plt.subplot(1, 2, 2)
    for snr_tr in train_snr_list:
        swipt_ckpt = f"checkpoints_swipt/deepjscc_swipt_snrtrain_{snr_tr}dB_rho0.3.pth"
        regular_ckpt = f"checkpoints/deepjscc_snrtrain_{snr_tr}dB.pth"
        
        if os.path.exists(swipt_ckpt):
            ckpt_path = swipt_ckpt
            label_prefix = "SWIPT"
        elif os.path.exists(regular_ckpt):
            ckpt_path = regular_ckpt
            label_prefix = "Regular"
        else:
            continue
        
        ckpt = torch.load(ckpt_path, map_location=device)
        model = DeepJSCC(latent_ch=latent_ch).to(device)
        model.load_state_dict(ckpt["state_dict"])
        
        energy_list = []
        for rho in rho_values:
            _, avg_energy = eval_swipt(model, test_loader, device, test_snr, rho, eta)
            energy_list.append(avg_energy)
        
        plt.plot(rho_values, energy_list, marker='o', label=f"{label_prefix} (SNR_train={snr_tr}dB)")
    
    plt.xlabel('Power Splitting Ratio (ρ)')
    plt.ylabel('Harvested Energy')
    plt.title(f'Harvested Energy vs ρ @ SNR_test={test_snr}dB')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    outpath = "results/swipt_re_curves.png"
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"\n[saved] {outpath}")
    plt.show()
    
    # Plot 3: Rate-Energy tradeoff (PSNR vs Energy)
    plt.figure(figsize=(8, 6))
    for snr_tr in train_snr_list:
        swipt_ckpt = f"checkpoints_swipt/deepjscc_swipt_snrtrain_{snr_tr}dB_rho0.3.pth"
        regular_ckpt = f"checkpoints/deepjscc_snrtrain_{snr_tr}dB.pth"
        
        if os.path.exists(swipt_ckpt):
            ckpt_path = swipt_ckpt
            label_prefix = "SWIPT"
        elif os.path.exists(regular_ckpt):
            ckpt_path = regular_ckpt
            label_prefix = "Regular"
        else:
            continue
        
        ckpt = torch.load(ckpt_path, map_location=device)
        model = DeepJSCC(latent_ch=latent_ch).to(device)
        model.load_state_dict(ckpt["state_dict"])
        
        psnr_list = []
        energy_list = []
        for rho in rho_values:
            avg_psnr, avg_energy = eval_swipt(model, test_loader, device, test_snr, rho, eta)
            psnr_list.append(avg_psnr)
            energy_list.append(avg_energy)
        
        plt.plot(energy_list, psnr_list, marker='o', label=f"{label_prefix} (SNR_train={snr_tr}dB)")
    
    plt.xlabel('Harvested Energy')
    plt.ylabel('PSNR (dB)')
    plt.title(f'Rate-Energy (R-E) Tradeoff @ SNR_test={test_snr}dB')
    plt.grid(True)
    plt.legend()
    
    outpath = "results/swipt_rate_energy_tradeoff.png"
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"[saved] {outpath}")
    plt.show()

if __name__ == "__main__":
    main()
