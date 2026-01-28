'''
A minimal example of training a Graph Convolutional Network (GCN) on the Cora dataset
for node classification using PyTorch Geometric.
'''

import os
import json

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge  


# -----------------------------
# 0) Reproducibility + output dir
# -----------------------------
torch.manual_seed(42)

OUT_DIR = "outputs_gcn"
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------
# 1) Load the Cora dataset
# -----------------------------
dataset = Planetoid(root="data_cora", name="Cora", transform=NormalizeFeatures())
data = dataset[0]  # Cora is a single graph (one Data object)

# We'll keep a copy of the original edges for robustness testing
edge_index_original = data.edge_index

print("=== Dataset info ===")
print("Nodes:", data.num_nodes)
print("Edges:", data.num_edges)
print("Node feature dim:", dataset.num_features)
print("Num classes:", dataset.num_classes)
print("Train nodes:", int(data.train_mask.sum()))
print("Val nodes:", int(data.val_mask.sum()))
print("Test nodes:", int(data.test_mask.sum()))


# -----------------------------
# 2) Define a simple 2-layer GCN
# -----------------------------
# Why these parts:
# - GCNConv: the "graph convolution" layer (neighbor aggregation)
# - Dropout: regularization (prevents overfitting on small graphs)
# - ReLU: non-linearity
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout_p=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout_p = dropout_p

    def forward(self, x, edge_index):
        # First graph conv
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Dropout only during training (F.dropout uses model.training flag)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # Second graph conv gives class logits per node
        x = self.conv2(x, edge_index)
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
model = GCN(in_dim=dataset.num_features, hidden_dim=16, out_dim=dataset.num_classes, dropout_p=0.5).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


# -----------------------------
# 3) Training loop (minimal)
# -----------------------------
# We train on nodes in train_mask only.
# Loss: cross-entropy on node classes
# Evaluation: accuracy on train/val/test masks
def accuracy_from_logits(logits, y_true):
    preds = logits.argmax(dim=1)
    correct = (preds == y_true).sum().item()
    return correct / y_true.numel()


best_val_acc = 0.0
best_state = None

print("\n=== Training ===")
for epoch in range(1, 201):
    model.train()

    # Forward pass
    out = model(data.x, edge_index_original)

    # Only compute loss on training nodes
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

    # Backprop + parameter update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Evaluate (no gradients needed)
    model.eval()
    with torch.no_grad():
        out_eval = model(data.x, edge_index_original)

        train_acc = accuracy_from_logits(out_eval[data.train_mask], data.y[data.train_mask])
        val_acc = accuracy_from_logits(out_eval[data.val_mask], data.y[data.val_mask])
        test_acc = accuracy_from_logits(out_eval[data.test_mask], data.y[data.test_mask])

    # Keep the best model by validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Print occasionally to keep output readable
    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | Train {train_acc:.4f} | Val {val_acc:.4f} | Test {test_acc:.4f}")

# Restore best validation checkpoint
model.load_state_dict(best_state)
model = model.to(device)

print(f"\nBest validation accuracy: {best_val_acc:.4f}")


# -----------------------------
# 4) Robustness test: edge dropout
# -----------------------------
# Idea:
# - Keep the trained model fixed.
# - Randomly drop edges with probability p.
# - Measure test accuracy as the graph becomes "more broken".
#
# dropout_edge(edge_index, p=...) returns a new edge_index with edges removed.
drop_ps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
test_accs = []

model.eval()
with torch.no_grad():
    for p in drop_ps:
        # training=False => deterministic behavior consistent with evaluation mode
        edge_index_dropped, _ = dropout_edge(edge_index_original, p=p, force_undirected=False, training=False)

        logits = model(data.x, edge_index_dropped)
        acc = accuracy_from_logits(logits[data.test_mask], data.y[data.test_mask])
        test_accs.append(acc)

        print(f"Edge dropout p={p:.1f} -> Test accuracy: {acc:.4f}")


# -----------------------------
# 5) Plot accuracy vs edge dropout
# -----------------------------
plt.figure(figsize=(7, 4))
plt.plot(drop_ps, test_accs, marker="o")
plt.xlabel("Edge dropout probability p")
plt.ylabel("Test accuracy")
plt.title("GCN robustness on Cora (accuracy vs edge dropout)")
plt.tight_layout()

plot_path = os.path.join(OUT_DIR, "accuracy_vs_edge_dropout.png")
plt.savefig(plot_path, dpi=200)
plt.close()


# -----------------------------
# 6) Save metrics to JSON
# -----------------------------
results = {
    "best_val_acc": float(best_val_acc),
    "drop_ps": drop_ps,
    "test_accs": [float(a) for a in test_accs],
}

with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

# Save trained model weights (optional but useful)
torch.save(model.state_dict(), os.path.join(OUT_DIR, "gcn_cora.pt"))

print("\n=== Saved outputs ===")
print("Plot:", plot_path)
print("Metrics:", os.path.join(OUT_DIR, "metrics.json"))
print("Model:", os.path.join(OUT_DIR, "gcn_cora.pt"))
