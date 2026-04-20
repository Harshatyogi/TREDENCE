"""
Self-Pruning Neural Network on CIFAR-10  (FINAL)
=================================================
Total Loss = CrossEntropyLoss + lambda * SparsityLoss
SparsityLoss = sum of all sigmoid(gate_scores)  [L1 norm of gates]

ROOT CAUSE FIX for Sparsity = 0%:
    gate_scores need to travel from +5 down to < -4.6 to make sigmoid < 0.01
    With a normal lr=1e-3 this takes thousands of steps and never reaches 0.
    FIX: gate_scores use a SEPARATE, MUCH HIGHER learning rate (0.5)
         while weights use normal lr (1e-3).
    This way L1 penalty drives unimportant gate_scores down fast -> real sparsity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════
# Part 1: PrunableLinear
# ══════════════════════════════════════════════════════

class PrunableLinear(nn.Module):
    """
    Linear layer with one learnable gate_score per weight.

    Forward:
        gates          = sigmoid(gate_scores)     in (0, 1)
        pruned_weights = weight * gates
        output         = F.linear(x, pruned_weights, bias)

    Sparsity mechanism:
        L1 penalty on gates pulls gate_scores down.
        gate_scores use a high LR (0.5) so they actually reach < -4.6
        which makes sigmoid < 0.01 (pruned).
        Important weights resist via strong classification gradients.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features))

        # Init gate_scores HIGH so all gates start open (sigmoid(5) = 0.993)
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), 5.0)
        )
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        gates          = torch.sigmoid(self.gate_scores)   # (0, 1)
        pruned_weights = self.weight * gates               # element-wise mask
        return F.linear(x, pruned_weights, self.bias)      # gradients flow to both

    def get_gates(self):
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores).cpu().flatten()

    def sparsity_loss(self):
        """L1 norm of gates = sum of all gate values (all positive after sigmoid)."""
        return torch.sigmoid(self.gate_scores).sum()


# ══════════════════════════════════════════════════════
# Part 2: Network
# ══════════════════════════════════════════════════════

class SelfPruningNet(nn.Module):
    """
    Feed-forward MLP using only PrunableLinear layers.
    Architecture: 3072 -> 1024 -> 512 -> 256 -> 128 -> 10
    """
    def __init__(self):
        super().__init__()

        self.fc1 = PrunableLinear(3072, 1024);  self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = PrunableLinear(1024,  512);  self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = PrunableLinear( 512,  256);  self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = PrunableLinear( 256,  128);  self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = PrunableLinear( 128,   10)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.bn1(self.fc1(x))); x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x))); x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x))); x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x

    def get_sparsity_loss(self):
        """
        SparsityLoss = sum of ALL gate values across ALL PrunableLinear layers.
        This is the L1 norm. Multiplied by lambda in the training loop.
        """
        total = torch.tensor(0.0)
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                total = total + m.sparsity_loss()
        return total

    def get_all_gates(self):
        gates = []
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                gates.append(m.get_gates())
        return torch.cat(gates)

    def compute_sparsity(self, threshold=1e-2):
        """% of gates below threshold = effectively pruned weights."""
        g = self.get_all_gates()
        return (g < threshold).float().mean().item() * 100.0

    def make_optimizer(self, weight_lr=1e-3, gate_lr=0.5, wd=1e-4):
        """
        KEY FIX: gate_scores need a much higher learning rate than weights.
        A gate_score must travel from +5 to < -4.6 to be considered pruned.
        With lr=1e-3 this never happens. With lr=0.5 it happens within epochs.
        """
        gate_params   = [p for n, p in self.named_parameters() if 'gate_scores' in n]
        weight_params = [p for n, p in self.named_parameters() if 'gate_scores' not in n]
        return optim.Adam([
            {'params': weight_params, 'lr': weight_lr, 'weight_decay': wd},
            {'params': gate_params,   'lr': gate_lr,   'weight_decay': 0.0},
        ])


# ══════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════

def get_loaders(batch_size=256):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10('./data', train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10('./data', train=False, download=True, transform=test_tf)

    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=2, pin_memory=True),
            DataLoader(test_ds,  batch_size=512, shuffle=False,
                       num_workers=2, pin_memory=True))


# ══════════════════════════════════════════════════════
# Part 3: Training Loop
# ══════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scheduler, lam, device):
    """
    Total Loss = CrossEntropyLoss + lambda * SparsityLoss
    SparsityLoss = sum of all gate values (L1 norm)
    """
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        cls_loss = F.cross_entropy(logits, labels)
        sp_loss  = model.get_sparsity_loss().to(device)

        # KEY formula: Total Loss = ClassificationLoss + lambda * SparsityLoss
        loss = cls_loss + lam * sp_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return correct / total * 100.0


def run_experiment(lam, train_loader, test_loader, device, epochs=30):
    print(f"\n{'='*60}")
    print(f"  Training  lambda = {lam}")
    print(f"{'='*60}")

    model     = SelfPruningNet().to(device)
    optimizer = model.make_optimizer(weight_lr=1e-3, gate_lr=0.5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[1e-3, 0.5],
        steps_per_epoch=len(train_loader),
        epochs=epochs, pct_start=0.1
    )

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, lam, device)

        if epoch % 5 == 0 or epoch == 1:
            acc = evaluate(model, test_loader, device)
            sp  = model.compute_sparsity()
            print(f"  Epoch {epoch:3d} | Loss {loss:.4f} | Acc {acc:.2f}% | Sparsity {sp:.1f}%")

    acc = evaluate(model, test_loader, device)
    sp  = model.compute_sparsity()
    print(f"\n  Final -> Acc: {acc:.2f}%   Sparsity: {sp:.1f}%")
    return model, acc, sp


# ══════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════

def plot_gate_distribution(model, lam, path='gate_distribution.png'):
    gates = model.get_all_gates().numpy()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(gates, bins=120, color='steelblue', edgecolor='white', linewidth=0.3)
    ax.axvline(0.01, color='red', linestyle='--', linewidth=1.5,
               label='Prune threshold (0.01)')
    ax.set_xlabel('Gate Value', fontsize=13)
    ax.set_ylabel('Count',      fontsize=13)
    ax.set_title(f'Gate Distribution — lambda={lam}', fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved -> {path}")


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice : {device}")
    print("gate_lr = 0.5 (high LR for gate_scores so they actually reach 0)")
    print("lambdas = [1e-4, 1e-3, 1e-2]\n")

    # ── Quick sanity check ────────────────────────────────────────────
    print("--- Sanity check (10 steps, lambda=1e-2) ---")
    _m   = SelfPruningNet()
    _opt = _m.make_optimizer(weight_lr=1e-3, gate_lr=0.5)
    sp0  = _m.compute_sparsity()
    for _ in range(30):
        _opt.zero_grad()
        x = torch.randn(16, 3, 32, 32)
        y = torch.randint(0, 10, (16,))
        (F.cross_entropy(_m(x), y) + 0.01 * _m.get_sparsity_loss()).backward()
        _opt.step()
    sp1 = _m.compute_sparsity()
    print(f"Sparsity: {sp0:.2f}% -> {sp1:.2f}%  ({'PASSED' if sp1 > sp0 else 'FAILED'})")
    assert sp1 > sp0, "Sparsity not moving — abort"
    del _m, _opt
    print()

    # ── Full training ─────────────────────────────────────────────────
    train_loader, test_loader = get_loaders(batch_size=256)

    lambdas    = [1e-4, 1e-3, 1e-2]   # low / medium / high sparsity pressure
    results    = []
    best_model = None
    best_lam   = lambdas[1]

    for lam in lambdas:
        model, acc, sp = run_experiment(lam, train_loader, test_loader,
                                        device, epochs=30)
        results.append({'lambda': lam, 'accuracy': acc, 'sparsity': sp})
        if lam == best_lam:
            best_model = model

    # ── Results table ─────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    print(f"  {'Lambda':<12} {'Test Acc (%)':<18} {'Sparsity (%)'}")
    print(f"  {'-'*50}")
    for r in results:
        print(f"  {r['lambda']:<12} {r['accuracy']:<18.2f} {r['sparsity']:.1f}")

    if best_model:
        plot_gate_distribution(best_model, best_lam)

    print("\nDone.")


if __name__ == '__main__':
    main()
