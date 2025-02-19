import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

dataset = Planetoid(root = './data', name = 'Cora', transform=T.NormalizeFeatures())
data = dataset[0]
train_mask = data.train_mask
test_mask = data.test_mask

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
model = GCN(in_channels=dataset.num_features, hidden_channels=128, out_channels=dataset.num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay=5e-4)

best_val_acc = 0
train_loss = []
train_acc = []

for epoch in range(200):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)

    loss = criterion(out[train_mask], data.y[train_mask])

    loss.backward()
    optimizer.step()

    train_loss.append(loss.item())

    _, pred = out.max(dim=1)
    correct = (pred[train_mask] == data.y[train_mask]).sum().item()
    acc = correct / train_mask.sum().item()
    train_acc.append(acc)

model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)

    prob = F.softmax(out[test_mask], dim=1).cpu().numpy()
    y_test = data.y[test_mask].cpu().numpy()

    auc = roc_auc_score(y_test, prob,multi_class='ovr', average='micro')
    print(f"Test ROC-AUC: {auc:.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(train_acc)
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
