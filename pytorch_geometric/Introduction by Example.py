# -*- coding: utf-8 -*-
"""
 @Time    : 2020/4/12
 @Author  : Han.yd
 @Site    : https://pytorch-geometric.readthedocs.io
 @File    : Introduction by Example.py
 @Software: PyCharm
"""
import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.data import DataLoader

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import ShapeNet

from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from torch_scatter import scatter_mean

"""
----------Data Handing of Graphs--------
data.x -> node features: [num_nodes, num_node_features]
data.edge_index -> COO format: triple tuple (u,v,c)
data.edge_attr  -> edge features: [num_edges, num_edge_features]
data.y: ground truth label -> [num_nodes, *]  e.g. nodes-level targets: [num_nodes, *]  graph-level targets: [1, *]  
data.pos: [num_nodes, num_dimensions] -> num_dimensions is the features of spatial position 
"""
edge_index = torch.tensor([[0,1,1,2],[1,0,2,1]], dtype = torch.long)
x = torch.tensor([[-1],[0],[1]], dtype = torch.float)
data = Data(x = x, edge_index = edge_index)


"""
-----------Common Benchmark Datasets------------ 
Cora, Citeseer, Punmed, etc
ENZYMES: Downloading http://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/ENZYMES.zip
Cora: Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.xxxx
ShapeNet: https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
"""
dataset = TUDataset(root = './data/ENZYMES' , name = 'ENZYMES')
dataset[0]: Data(edge_index=[2, 168], x=[37, 3], y=[1])
# The first graph: 168 edges, 37 nodes, every node has three features
perm = torch.randperm(len(dataset))
dataset = dataset[perm]
dataset = Planetoid(root = './data/Cora', name = 'Cora')


"""
-----------Mini-batches------------ 
"""
dataset = TUDataset(root = './data/ENZYMES' , name = 'ENZYMES')
loader = DataLoader(dataset, batch_size = 32, shuffle = True)
for i, data in enumerate(loader):
    # In the first batch, there are 1047 nodes (belong to the 32 graphs (i.e.,32 batches)), [1047, 3]
    print(data.x.size())


"""
-----------Data Transforms------------ 
"""
# Only datasets
dataset = ShapeNet(root = './data/ShapeNet', categories = ['Airplane'])
print(dataset[0])   # >>> Data(pos=[2518, 3], y=[2518])
# Constucting the Graph with neighbor
dataset = ShapeNet(root = './data/ShapeNet', categories = ['Airplane'],
                   pre_transform = T.KNNGraph(k = 6)) # >>> Data(edge_index=[2, 15108], pos=[2518, 3], y=[2518])
# translate each node position by a small number:
dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'],
                    pre_transform=T.KNNGraph(k=6),
                    transform=T.RandomTranslate(0.01))


"""
-----------Learning Methods on Graphs---------------
Let's implement a two-layer GCN
"""
# Load the datasets: a specific datasets format
dataset = Planetoid(root = './data/Cora', name = 'Cora')  # Only one Graph
# datasets中含有的一些属性: dataset.num_classes, dataset.num_node_features, dataset.num_edge_features, dataset.num_features

# Construct the model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net , self).__init__() # torch.nn.module没有去实现__init__，因此这里不需要传参数
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)  # 指定投入数据格式
        x = F.relu(x)
        x = F.dropout(x, training = self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim = 1)

# Preparation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device) # 另外model.cuda()和tensor.cuda()的作用是不一样的
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 5e-4)

# Start to train
model.train() # -> self.training = True (model.eval() -> self.training = False)

for epoch in range(200):
    optimizer.zero_grad() # 梯度需要清0，因为每一次梯度的累加的
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward() # 如果是标量的话可以执行这样的操作，如果是向量的话，需要在backward中传值
    optimizer.step()

# Start to test
model.eval()
_, pred = model(data).max(dim = 1) # 返回值和下标
correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct/data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))




