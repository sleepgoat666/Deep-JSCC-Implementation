import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import SemanticSWIPTDeepJSCC


def train_one_epoch(model, loader, opt, device, snr_train, e_min, beta, eta, h_ir, h_er):
    model.train()
    total_loss = 0.0
    total_task = 0.0
    total_pen = 0.0
    total_correct = 0
    total_samples = 0
    total_energy_sum = 0.0
    total_violation = 0

    for x, label in loader:
        x = x.to(device)
        label = label.to(device)

        logits, harvested_energy = model(x, snr_train, eta=eta, h_ir=h_ir, h_er=h_er)
        task_loss = F.cross_entropy(logits, label)
        energy_penalty = F.relu(e_min - harvested_energy).mean()
        loss = task_loss + beta * energy_penalty

        opt.zero_grad()
        loss.backward()
        opt.step()

        bs = x.size(0)
        pred = logits.argmax(dim=1)
        total_correct += (pred == label).sum().item()
        total_samples += bs

        total_loss += loss.item() * bs
        total_task += task_loss.item() * bs
        total_pen += energy_penalty.item() * bs
        total_energy_sum += harvested_energy.sum().item()
        total_violation += (harvested_energy < e_min).sum().item()

    return {
        "loss": total_loss / total_samples,
        "task_loss": total_task / total_samples,
        "energy_penalty": total_pen / total_samples,
        "accuracy": total_correct / total_samples,
        "avg_harvested_energy": total_energy_sum / total_samples,
        "energy_violation_rate": total_violation / total_samples,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[device]", device, flush=True)

    # training hyperparameters
    snr_train = 7
    epochs = 20
    latent_ch = 8
    lr = 1e-3
    batch_size = 128
    e_min = 0.2
    beta = 1.0
    eta = 0.5
    h_ir = 1.0
    h_er = 1.0

    tfm = transforms.Compose([transforms.ToTensor()])
    os.makedirs("./data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    model = SemanticSWIPTDeepJSCC(latent_ch=latent_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        metrics = train_one_epoch(
            model,
            train_loader,
            opt,
            device,
            snr_train=snr_train,
            e_min=e_min,
            beta=beta,
            eta=eta,
            h_ir=h_ir,
            h_er=h_er,
        )
        print(
            f"Epoch {epoch:02d} | "
            f"loss={metrics['loss']:.6f} | "
            f"task_loss={metrics['task_loss']:.6f} | "
            f"energy_penalty={metrics['energy_penalty']:.6f} | "
            f"accuracy={metrics['accuracy']:.4f} | "
            f"avg_harvested_energy={metrics['avg_harvested_energy']:.6f} | "
            f"energy_violation_rate={metrics['energy_violation_rate']:.4f}",
            flush=True,
        )

    ckpt_path = f"checkpoints/semantic_swipt_snr{snr_train}_emin{e_min}_beta{beta}.pth"
    torch.save(
        {
            "snr_train": snr_train,
            "E_min": e_min,
            "beta": beta,
            "eta": eta,
            "h_ir": h_ir,
            "h_er": h_er,
            "state_dict": model.state_dict(),
        },
        ckpt_path,
    )
    print("[saved]", ckpt_path, flush=True)


if __name__ == "__main__":
    main()
