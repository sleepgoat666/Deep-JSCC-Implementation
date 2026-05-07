import csv
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import SemanticSWIPTDeepJSCC


@torch.no_grad()
def eval_semantic(model, loader, device, snr_db, e_min, eta, h_ir, h_er):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_energy = 0.0
    total_violation = 0

    for x, label in loader:
        x = x.to(device)
        label = label.to(device)

        logits, harvested_energy = model(x, snr_db, eta=eta, h_ir=h_ir, h_er=h_er)
        pred = logits.argmax(dim=1)

        bs = x.size(0)
        total_correct += (pred == label).sum().item()
        total_samples += bs
        total_energy += harvested_energy.sum().item()
        total_violation += (harvested_energy < e_min).sum().item()

    return {
        "accuracy": total_correct / total_samples,
        "avg_harvested_energy": total_energy / total_samples,
        "energy_violation_rate": total_violation / total_samples,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[device]", device, flush=True)

    checkpoint_path = "checkpoints/semantic_swipt_snr7_emin0.2_beta1.0.pth"
    snr_test_list = [1, 4, 7, 13, 19]
    latent_ch = 8

    tfm = transforms.Compose([transforms.ToTensor()])
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    test_loader = DataLoader(
        testset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = SemanticSWIPTDeepJSCC(latent_ch=latent_ch).to(device)
    model.load_state_dict(ckpt["state_dict"])

    e_min = ckpt.get("E_min", 0.2)
    eta = ckpt.get("eta", 0.5)
    h_ir = ckpt.get("h_ir", 1.0)
    h_er = ckpt.get("h_er", 1.0)

    rows = []
    for snr_te in snr_test_list:
        metrics = eval_semantic(
            model,
            test_loader,
            device,
            snr_db=snr_te,
            e_min=e_min,
            eta=eta,
            h_ir=h_ir,
            h_er=h_er,
        )
        row = {
            "snr_db": snr_te,
            "accuracy": metrics["accuracy"],
            "avg_harvested_energy": metrics["avg_harvested_energy"],
            "energy_violation_rate": metrics["energy_violation_rate"],
        }
        rows.append(row)
        print(
            f"SNR={snr_te:>2} dB | "
            f"accuracy={row['accuracy']:.4f} | "
            f"avg_harvested_energy={row['avg_harvested_energy']:.6f} | "
            f"energy_violation_rate={row['energy_violation_rate']:.4f}",
            flush=True,
        )

    os.makedirs("results", exist_ok=True)

    csv_path = "results/semantic_swipt_eval.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["snr_db", "accuracy", "avg_harvested_energy", "energy_violation_rate"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print("[saved]", csv_path)

    snrs = [r["snr_db"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    energies = [r["avg_harvested_energy"] for r in rows]

    plt.figure()
    plt.plot(snrs, accs, marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.title("Semantic-SWIPT-DeepJSCC: Accuracy vs SNR")
    plt.grid(True)
    acc_plot_path = "results/semantic_swipt_accuracy_vs_snr.png"
    plt.savefig(acc_plot_path, dpi=200, bbox_inches="tight")
    print("[saved]", acc_plot_path)

    plt.figure()
    plt.plot(snrs, energies, marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Average Harvested Energy")
    plt.title("Semantic-SWIPT-DeepJSCC: Harvested Energy vs SNR")
    plt.grid(True)
    energy_plot_path = "results/semantic_swipt_energy_vs_snr.png"
    plt.savefig(energy_plot_path, dpi=200, bbox_inches="tight")
    print("[saved]", energy_plot_path)


if __name__ == "__main__":
    main()
