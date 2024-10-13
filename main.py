import torch
import torch.optim as optim
import torch.nn.functional as F

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

from data import loadData
from model import GCN

# Load Data
dataCiteseer = loadData()
data = dataCiteseer[0]

print(f'dataCiteseer: {dataCiteseer}')
print(f'Number of graphs: {len(dataCiteseer)}')
print(f'Number of features: {dataCiteseer.num_features}')
print(f'Number of classes: {dataCiteseer.num_classes}')
print(f'Data object: {data}')
print('----------------------------------------------------------\n')

print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Number of validation nodes: {data.val_mask.sum()}')
print(f'Number of test nodes: {data.test_mask.sum()}')
print(f'Node features shape: {data.x.shape}')
print(f'Edge index shape: {data.edge_index.shape}')
print('----------------------------------------------------------\n')





# # Visualize data
# G = to_networkx(data=data)

# plt.figure(figsize=(10, 10))
# nx.draw(G, node_size=50, node_color=data.y, cmap=plt.get_cmap('cool'), with_labels=False)
# plt.show()

# Model
model = GCN(dataCiteseer.num_features, dataCiteseer.num_classes)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Testing Function
def test():
    model.eval()
    logits = model(data)
    pred = logits.argmax(dim=1)
    correct = pred[data.test_mask] == data.y[data.test_mask]
    acc = int(correct.sum())/int(data.test_mask.sum())
    return acc

# Training Epoch
for epoch in range(1000):
    loss = train()
    if epoch%10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

acc = test()
print(f'Accuracy: {acc*100:.2f}%')
