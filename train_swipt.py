# -*- coding: utf-8 -*-
"""
SWIPT-enabled Deep JSCC training script.
Trains Deep JSCC with Simultaneous Wireless Information and Power Transfer.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import DeepJSCC
from metrics import psnr

@torch.no_grad()
def eval_psnr_swipt(model, loader, device, snr_db, rho, eta):
    """
    Evaluate PSNR for SWIPT model.
    
    Args:
        model: DeepJSCC model
        loader: Data loader
        device: Device (cuda/cpu)
        snr_db: SNR in dB
        rho: Power splitting ratio
        eta: Energy harvesting efficiency
    
    Returns:
        Average PSNR
    """
    model.eval()
    acc = 0.0
    for x, _ in loader:
        x = x.to(device)
        xhat, _ = model(x, snr_db, use_swipt=True, rho=rho, eta=eta)
        acc += psnr(x, xhat).item() * x.size(0)
    return acc / len(loader.dataset)

def train_one_epoch_swipt(model, loader, opt, device, snr_train, rho, eta, lambda_energy):
    """
    Train one epoch with SWIPT.
    
    Args:
        model: DeepJSCC model
        loader: Data loader
        opt: Optimizer
        device: Device (cuda/cpu)
        snr_train: Training SNR (fixed or tuple for range)
        rho: Power splitting ratio for energy harvesting
        eta: Energy harvesting efficiency
        lambda_energy: Weight for energy harvesting term in loss
    
    Returns:
        Average loss, average MSE, average harvested energy
    """
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_energy = 0.0
    fixed = not isinstance(snr_train, (tuple, list))

    for x, _ in loader:
        x = x.to(device)

        if fixed:
            snr_db = float(snr_train)
        else:
            snr_db = float(torch.empty(1).uniform_(snr_train[0], snr_train[1]).item())

        xhat, e_harvested = model(x, snr_db, use_swipt=True, rho=rho, eta=eta)
        
        # Reconstruction loss (MSE)
        mse_loss = F.mse_loss(xhat, x)
        
        # Energy harvesting term
        energy_term = e_harvested.mean()
        
        # Combined loss: MSE - lambda * harvested_energy
        # We want to minimize MSE and maximize energy, so we subtract energy term
        loss = mse_loss - lambda_energy * energy_term

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item() * x.size(0)
        total_mse += mse_loss.item() * x.size(0)
        total_energy += energy_term.item() * x.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_mse / n, total_energy / n

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[device]", device, flush=True)

    # SWIPT parameters
    train_snr_list = [7, 13, 19]  # Train at higher SNRs for SWIPT
    epochs = 5
    latent_ch = 8
    lr = 1e-3
    
    # SWIPT-specific parameters
    rho = 0.3  # Power splitting ratio (30% for energy harvesting)
    eta = 0.8  # Energy harvesting efficiency (80%)
    lambda_energy = 0.01  # Weight for energy term in loss

    tfm = transforms.Compose([transforms.ToTensor()])
    os.makedirs("./data", exist_ok=True)
    os.makedirs("checkpoints_swipt", exist_ok=True)

    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    testset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)

    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0,
                              pin_memory=(device == "cuda"))
    test_loader  = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0,
                              pin_memory=(device == "cuda"))

    for snr_tr in train_snr_list:
        print(f"\n===== Train SWIPT @ SNR_train={snr_tr} dB, rho={rho}, eta={eta}, lambda={lambda_energy} =====", flush=True)

        model = DeepJSCC(latent_ch=latent_ch).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            tr_loss, tr_mse, tr_energy = train_one_epoch_swipt(
                model, train_loader, opt, device, snr_train=snr_tr,
                rho=rho, eta=eta, lambda_energy=lambda_energy
            )
            p = eval_psnr_swipt(model, test_loader, device, snr_db=snr_tr, rho=rho, eta=eta)
            print(f"Epoch {epoch:02d} | loss={tr_loss:.6f} | mse={tr_mse:.6f} | energy={tr_energy:.6f} | PSNR@{snr_tr}dB={p:.2f}", flush=True)

        ckpt_path = f"checkpoints_swipt/deepjscc_swipt_snrtrain_{snr_tr}dB_rho{rho}.pth"
        torch.save({
            "snr_train": snr_tr,
            "rho": rho,
            "eta": eta,
            "lambda_energy": lambda_energy,
            "state_dict": model.state_dict()
        }, ckpt_path)
        print("[saved]", ckpt_path, flush=True)

if __name__ == "__main__":
    main()
