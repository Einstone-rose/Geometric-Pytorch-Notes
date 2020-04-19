# -*- coding: utf-8 -*-
"""
 @Time    : 2020/4/12
 @Author  : Han.yd
 @Site    : https://pytorch-geometric.readthedocs.io
 @File    : Creating Message Passing Networks.py
 @Software: PyCharm
"""
import torch
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.datasets import Planetoid
import inspect
from collections import OrderedDict
import torch
from torch_sparse import SparseTensor
from torch_scatter import gather_csr, scatter, segment_csr
__all__ = ['GCNconv', 'Edgeconv']

class GCNconv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNconv, self).__init__(aggr = 'add')
        self.linear = torch.nn.Linear(in_channels, out_channels)
    def forward(self, x, edge_idx):
        # x: [N, in_channels], edge_idx: [2, E]
        # step1: add self-loops to the adjacnecy matrix
        edge_idx, _  = add_self_loops(edge_idx, num_nodes = x.size(0))
        # Step 2: Linearly transform node feature matrix.
        x = self.linear(x)
        # Step 3: Compute normalization
        row, col = edge_idx

        deg = degree(row, x.size(0), dtype = x.dtype)  # -> enter into  [1, 2078]
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]   # -> 取出对应idx对应的数值
        # Step 4-6: Start to propagate messages
        return self.propagate(edge_idx, size=(x.size(0), x.size(0)), x=x, norm=norm)
    def message(self, x_j, norm): # 进行重写
        # print(norm.size()), print(x_j.size())
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out
    def __collect__(self, edge_index, size, mp_type, kwargs):
        # self.flow = source_to_target, kwargs: {'x':.., 'norm':...}
        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        # (1,0)
        ij = {'_i': i, '_j': j}
        out = {}
        # print(kwargs['x'].size())
        for arg in self.__user_args__:   # self.__user_args__: {'x_j', 'norm'}
            if arg[-2:] not in ij.keys():
                out[arg] = kwargs.get(arg, inspect.Parameter.empty)
            else:
                idx = ij[arg[-2:]] # 0
                data = kwargs.get(arg[:-2], inspect.Parameter.empty) # [2708, 16] --> ['x'] ?????????????

                if data is inspect.Parameter.empty: # False
                    out[arg] = data
                    continue

                if isinstance(data, tuple) or isinstance(data, list):
                    assert len(data) == 2
                    self.__set_size__(size, 1 - idx, data[1 - idx])
                    data = data[idx]

                if not torch.is_tensor(data):
                    out[arg] = data
                    continue

                self.__set_size__(size, idx, data)
                if mp_type == 'edge_index':
                    # arg, idx, self.node_dim = (x_j,0,0), data: [2708, 16]
                    out[arg] = data.index_select(self.node_dim, edge_index[idx])
                    # [E, out_channels]
                elif mp_type == 'adj_t' and idx == 1:
                    rowptr = edge_index.storage.rowptr()
                    for _ in range(self.node_dim):
                        rowptr = rowptr.unsqueeze(0)
                    out[arg] = gather_csr(data, rowptr)
                elif mp_type == 'adj_t' and idx == 0:
                    col = edge_index.storage.col()
                    out[arg] = data.index_select(self.node_dim, col)

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        if mp_type == 'edge_index':
            out['edge_index_j'] = edge_index[j]
            out['edge_index_i'] = edge_index[i]
            out['index'] = out['edge_index_i']
        elif mp_type == 'adj_t':
            out['adj_t'] = edge_index
            out['edge_index_i'] = edge_index.storage.row()
            out['edge_index_j'] = edge_index.storage.col()
            out['index'] = edge_index.storage.row()
            out['ptr'] = edge_index.storage.rowptr()
            out['edge_attr'] = edge_index.storage.value()

        out['size_j'] = size[j]
        out['size_i'] = size[i]
        out['dim_size'] = out['size_i']
        return out
    # 自己重写和调试
    def propagate(self, edge_index, size=None, **kwargs):
        # We need to distinguish between the old `edge_index` format and the
        # new `torch_sparse.SparseTensor` format.
        mp_type = self.__get_mp_type__(edge_index) # -> return 'edge_index'

        if mp_type == 'adj_t' and self.flow == 'target_to_source':
            raise ValueError(
                ('Flow direction "target_to_source" is invalid for message '
                 'propagation based on `torch_sparse.SparseTensor`. If you '
                 'really want to make use of a reverse message passing flow, '
                 'pass in the transposed sparse tensor to the message passing '
                 'module, e.g., `adj.t()`.'))
        # Size is a tuple
        if mp_type == 'edge_index':
            if size is None:
                size = [None, None]
            elif isinstance(size, int):
                size = [size, size]
            elif torch.is_tensor(size):
                size = size.tolist()
            elif isinstance(size, tuple):
                size = list(size) # -> [size(0),size(0)]
        elif mp_type == 'adj_t':
            size = list(edge_index.sparse_sizes())[::-1]

        assert isinstance(size, list)
        assert len(size) == 2

        # We collect all arguments used for message passing in `kwargs`.
        kwargs = self.__collect__(edge_index, size, mp_type, kwargs)   # --> this is the key code

        if isinstance(kwargs, dict):
            for key, value in kwargs.items():
                print("{0}:{1}".format(key, value))
        else:
            print("The error is found...")
            exit()
        # intermediate output:
        # The parameters of kwargs (dict):
        # x_j:tensor([[-0.0489, -0.0553,  0.0311,  ..., -0.0562,  0.0348,  0.0362],
        #             [-0.0489, -0.0553,  0.0311,  ..., -0.0562,  0.0348,  0.0362],
        #             [-0.0489, -0.0553,  0.0311,  ..., -0.0562,  0.0348,  0.0362],
        #             ...,
        #             [-0.0224, -0.1631, -0.0529,  ..., -0.0209,  0.0061,  0.0461],
        #             [-0.0477,  0.0212, -0.0367,  ..., -0.1194,  0.0508,  0.0352],
        #             [-0.0084, -0.0034,  0.0068,  ..., -0.0378,  0.0792, -0.0542]],
        #     grad_fn=<IndexSelectBackward>)
        # edge_index_j:tensor([   0,    0,    0,  ..., 2705, 2706, 2707])
        # edge_index_i:tensor([ 633, 1862, 2582,  ..., 2705, 2706, 2707])
        # index:tensor([ 633, 1862, 2582,  ..., 2705, 2706, 2707])
        # size_j:2708
        # size_i:2708
        # dim_size:2708


        # mp_type: edge_index , self__fuse__: True
        # Try to run `message_and_aggregate` first and see if it succeeds:
        if mp_type == 'adj_t' and self.__fuse__ is True:
            msg_aggr_kwargs = self.__distribute__(self.__msg_aggr_params__,
                                                  kwargs)
            out = self.message_and_aggregate(**msg_aggr_kwargs)
            if out == NotImplemented:
                self.__fuse__ = False

        # Otherwise, run both functions in separation.
        if mp_type == 'edge_index' or self.__fuse__ is False:

            # self.__msg_params__ --> OrderedDict([('x_j', <Parameter "x_j">), ('norm', <Parameter "norm">)])
            msg_kwargs = self.__distribute__(self.__msg_params__, kwargs) # -> 根据kwargs中出现query去self.__msg_aggr_params__查看相应的参数
            out = self.message(**msg_kwargs)

            # OrderedDict([('index', <Parameter "index">),
            # ('ptr', <Parameter "ptr=None">),
            # ('dim_size', <Parameter "dim_size=None">)])
            aggr_kwargs = self.__distribute__(self.__aggr_params__, kwargs)
            out = self.aggregate(out, **aggr_kwargs)

        update_kwargs = self.__distribute__(self.__update_params__, kwargs)
        out = self.update(out, **update_kwargs)

        return out
# Call for paper: https://arxiv.org/abs/1801.07829v1 (DGCNN -> point clouds)
class Egdeconv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Egdeconv, self).__init__(aggr = 'max')
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels)
                       )
    def forward(self, x, edge_idx):
        # x has shape [N, in_channels], edge_idx: [2,E]
        return self.propagate(edge_idx, size = (x.size(0), x.size(0)), x = x)

    def message(self, x_i, x_j):
        # x_i: [E, in_channels], E代表了邻边的数量，即邻边的节点
        # x_j: [E, in_channels]
        # 既考虑了局部结构，也考虑了全局结构
        tmp = torch.cat([x_i, x_j - x_i], dim = 1)
        return self.mlp(tmp)
if __name__=='__main__':
    # Construct the datasets
    data = Planetoid(root = "./data/Cora", name = "Cora")
    # Construct the model
    net = GCNconv(data[0].x.size(1), 16)
    out = net(data[0].x, data[0].edge_index)


