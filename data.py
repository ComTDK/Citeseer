from torch_geometric.datasets import Planetoid

def loadData(): 
    citeseerData = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    return citeseerData