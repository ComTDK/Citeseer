import torch
import torch.optim as optim
import torch.nn.functional as F

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

from data import loadData


# Load Data
dataCiteseer = loadData()

# Visualize data
G = to_networkx(data=dataCiteseer[0])

plt.figure(figsize=(10, 10))
nx.draw(G, node_size=50, node_color=dataCiteseer[0].y, cmap=plt.get_cmap('cool'), with_labels=False)
plt.show()