import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from sklearn.neighbors import LSHForest
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected, degree, negative_sampling, to_dense_adj, to_edge_index
from torch_geometric.nn import GCNConv
from torch_geometric.utils.num_nodes import maybe_num_nodes
import seaborn as sns
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from sklearn.manifold import TSNE
import bhtsne

try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None

def to_sparse_tensor(edge_index, num_nodes):
    return SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    ).to(edge_index.device)


class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers=2, dropout=0.5, use_bn=False):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        bn = nn.BatchNorm1d if use_bn else nn.Identity
        
        for i in range(n_layers):
            first_dim = in_dim if i == 0 else hid_dim
            second_dim = out_dim if i == n_layers - 1 else hid_dim

            self.convs.append(GCNConv(first_dim, second_dim))
            self.bns.append(bn(second_dim))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for bn in self.bns:
            if not isinstance(bn, nn.Identity):
                bn.reset_parameters()

    def forward(self, x, edge_index):
        edge_sparse = to_sparse_tensor(edge_index, x.size(0))
        
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_sparse)
            x = self.bns[i](x)
            x = self.activation(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_sparse)
        x = self.bns[-1](x)
        x = self.activation(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers=2, dropout=0.5, use_bn=False):
        super().__init__()
        
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        bn = nn.BatchNorm1d if use_bn else nn.Identity
        
        for i in range(n_layers):
            first_dim = in_dim if i == 0 else hid_dim
            second_dim = out_dim if i == n_layers - 1 else hid_dim

            self.lins.append(nn.Linear(first_dim, second_dim))
            self.bns.append(bn(second_dim))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

        for bn in self.bns:
            if not isinstance(bn, nn.Identity):
                bn.reset_parameters()

    def forward(self, x):
        
        for i, lin in enumerate(self.lins[:-1]):
            x = self.dropout(x)
            x = lin(x)
            x = self.bns[i](x)
            x = self.activation(x)
        x = self.dropout(x)
        x = self.lins[-1](x)
        x = self.bns[-1](x)
        x = self.activation(x)
        return x

class Masker(nn.Module):
    def __init__(self, p=0.5, undirected=True, walks_per_node=1, walk_length=3, num_nodes=None):
        super().__init__()
        self.p = p
        self.undirected = undirected
        
        self.walks_per_node = walks_per_node
        self.walk_length = walk_length
        self.num_nodes = num_nodes
    
    def mask_edges(self, edge_index, p, walks_per_node=1, walk_length=3, num_nodes=None, training=True):
        
        random_walk = torch.ops.torch_cluster.random_walk
        
        edge_mask = edge_index.new_ones(edge_index.shape[1], dtype=torch.bool)
        
        if not training or p == 0.0:
            return edge_index, edge_mask

        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        row, col = edge_index
        start = torch.randperm(num_nodes, device=edge_index.device)[:round(num_nodes*p)].repeat(walks_per_node)
        
        deg = degree(row, num_nodes=num_nodes)
        rowptr = row.new_zeros(num_nodes + 1)
        torch.cumsum(deg, 0, out=rowptr[1:])
        n_id, e_id = random_walk(rowptr, col, start, walk_length, 1.0, 1.0)
        e_id = e_id[e_id != -1].view(-1)  # filter illegal edges
        edge_mask[e_id] = False

        return edge_index[:, edge_mask], edge_index[:, ~edge_mask]

    def mask_edge2(self, edge_index, p=0.7):
        if p < 0. or p > 1.:
            raise ValueError(f'Mask probability has to be between 0 and 1 '
                            f'(got {p}')    
        e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        mask = torch.full_like(e_ids, p, dtype=torch.float32)
        mask = torch.bernoulli(mask).to(torch.bool)
        return edge_index[:, ~mask], edge_index[:, mask]

    def forward(self, edge_index):
        # remaining_edges, masked_edges = self.mask_edges(edge_index, self.p, 
        #                                         self.walks_per_node,
        #                                         walk_length=self.walk_length,
        #                                         num_nodes=self.num_nodes
        #                                         )
        remaining_edges, masked_edges = self.mask_edge2(edge_index, self.p)
        if self.undirected:
            remaining_edges = to_undirected(remaining_edges)
            
        return remaining_edges, masked_edges

class Adj_decoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim=1, dropout=0.3, mode='cat'):
        super().__init__()
        # self.lin = nn.Linear(in_dim*2, hid_dim)
        self.mode = mode
        self.projector = nn.Sequential(
            nn.Dropout(dropout)
          , nn.Linear(in_dim*3, hid_dim)
          , nn.ReLU()
          , nn.Linear(hid_dim, out_dim)
        )
    
    def forward(self, h, p, edges):
        u, v = edges[0], edges[1]
        h_cat = torch.cat([h, p], dim=-1)
        # h_cat = h
        # h_cat = self.lin(h_cat)
        if self.mode == 'dot':
            edge_h = torch.cat([h[u] * p[v], h[v] * p[u]], dim=-1)
        elif self.mode == 'cat':  
            edge_h = torch.cat([h[u] * p[v], h[v] * p[u], h[u] * h[v]], dim=-1)
        s1 = self.projector(edge_h)
        return s1
    
    @torch.no_grad()
    def get_recon_adj(self, h, p, h_m, edges):
        u, v = edges[0], edges[1]
        edges_n = negative_sampling(edges, num_neg_samples=edges.shape[1])
        u_neg, v_neg = edges_n[0], edges_n[1]
        
        n_nodes = h.shape[0]
        s1 = self.projector(
            torch.cat([h[u] * p[v], h[v] * p[u], h[u] * h[v]], dim=-1)
        )
        s2 = self.projector(
            torch.cat([h[u_neg] * p[v_neg], h[v_neg] * p[u_neg], h[u_neg] * h[v_neg]], dim=-1)
        )
        recon_edges = s1
        recon_negs = s2
        
        A = torch.zeros([n_nodes, n_nodes]).to(h)
        A[u, v] = recon_edges.sigmoid().squeeze()
        
        i = torch.arange(n_nodes)
        diag = self.projector(
            torch.cat([h[i] * p[i], h[i] * p[i], h[i] * h[i]], dim=-1)
        )
        A[torch.arange(n_nodes), torch.arange(n_nodes)] = torch.sigmoid(diag).squeeze()
        
        return A

class LPGRL(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, batch_size, n_nodes, n_feats, use_bn_g=False, use_bn_m=False, adj_dropout=0.5, p=0.7, use_intr=False):
        super(LPGRL, self).__init__()
        self.batch_size = batch_size
        self.use_intr = use_intr
        self.masker = Masker(p=p)
        
        self.gnn_encoder = GCN(in_dim, hid_dim, out_dim, n_layers, use_bn=use_bn_g)
        self.mlp_encoder = MLP(in_dim, hid_dim, out_dim, n_layers, use_bn=use_bn_m)
        if not self.use_intr:
            self.P = nn.Parameter(torch.randn(n_nodes, out_dim), requires_grad=True)
        # self.lin = nn.Linear(in_dim, out_dim)
   
        self.adj_decoder = Adj_decoder(out_dim, out_dim, dropout=adj_dropout)
        
        self.proj_head_1 = nn.Sequential(nn.Linear(out_dim, out_dim))
        self.proj_head_2 = nn.Sequential(nn.Linear(out_dim, out_dim))
        self.proj_head_p = nn.Sequential(nn.Linear(out_dim, out_dim))
        

     def set_mask_knn(self, x, edge_index, k, name, metric='cosine', use_lsh=False):
        if k != 0:
            path = f'./data/knn'
            file_name = path + f'/{name}_{k}.npz'
            if os.path.exists(file_name):
                knn = sp.load_npz(file_name)
            else:
                if use_lsh:
                    lshf = LSHForest()
                    lshf.fit(x.cpu().detach().numpy()) 
                    knn = lshf.kneighbors_graph(x.cpu().detach().numpy(), k, mode='distance')
                    knn = knn.toarray()
                else:
                    knn = kneighbors_graph(x.cpu().detach().numpy(), k, metric=metric)
                    sp.save_npz(file_name, knn)
            knn = torch.tensor(knn.toarray()) + torch.eye(x.shape[0])
        else:
            knn = torch.eye(x.shape[0])
        self.pos_mask = knn
        self.neg_mask = 1 - self.pos_mask
        self.edge_pos_mask = to_dense_adj(edge_index, max_num_nodes=x.shape[0]).squeeze()
        self.edge_neg_mask = 1 - self.edge_pos_mask - torch.eye(x.shape[0]).to(x)
    
    @torch.no_grad()
    def get_embedding(self, x, edge_index):
        h_g = self.gnn_encoder(x, edge_index)
        h_m = self.mlp_encoder(x)
        
        projg = self.proj_head_1(h_g)
        projm = self.proj_head_2(h_m)
        projp = self.proj_head_p(self.P)


        return torch.cat((h_g, h_m), dim=1), h_g, self.P, h_m, projg, projm, projp
    
    def forward(self, x, edge_index, alpha=0.5):
        
        remaining_edges, masked_edges = self.masker(edge_index)
        neg_edges = negative_sampling(edge_index, num_neg_samples=masked_edges.size(1))
        
        h_g = self.gnn_encoder(x, remaining_edges)
        h_m = self.mlp_encoder(x)
        if self.use_intr == True:
            self.P = h_m
            
        proj1 = self.proj_head_1(h_g)
        proj2 = self.proj_head_2(h_m)
        
        projp = self.proj_head_p(self.P)
   
        pos_edge = self.adj_decoder(
            h_g, self.P, masked_edges
        )
        neg_edge = self.adj_decoder(
            h_g, self.P, neg_edges
        )
        
        loss1 = self.adj_recon_loss(pos_edge, neg_edge)
        loss2 = self.str_cl_loss(projp, proj2)
        loss3 = self.sem_cl_loss(proj1, proj2)

        return alpha * (loss1 + loss2) + (1 - alpha) * loss3

    
    def split_batch(self, init_list, batch_size):
        groups = zip(*(iter(init_list),) * batch_size)  
        end_list = [list(i) for i in groups]
        count = len(init_list) % batch_size  
        end_list.append(init_list[-count:]) if count != 0 else end_list 
        return end_list
    
    
    def adj_recon_loss(self, pos_out, neg_out):
        pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
        neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
        return pos_loss + neg_loss
    
    
    def str_cl_loss(self, p, h, temperature=0.2):
        pos_mask = self.edge_pos_mask
        neg_mask = self.edge_neg_mask
        
        nnodes = p.shape[0]
        if (self.batch_size == 0) or (self.batch_size > nnodes):  # 不划分batch
            loss = self.infonce(p, h, pos_mask, neg_mask, temperature)
        else:
            node_idxs = list(range(nnodes))
            random.shuffle(node_idxs)
            batches = self.split_batch(node_idxs, self.batch_size)
            loss = 0
            for b in batches:
                weight = len(b) / nnodes
                loss_ = self.infonce(p[b], h[b], pos_mask[:,b][b,:], neg_mask[:,b][b,:], temperature)
                loss += loss_ * weight
        return loss
            
    
    def sem_cl_loss(self, z1, z2, temperature=0.2):
        pos_mask = self.pos_mask
        neg_mask = self.neg_mask

        nnodes = z1.shape[0]  # 节点数
        if (self.batch_size == 0) or (self.batch_size > nnodes):  # 不划分batch
            loss_0 = self.infonce(z1, z2, pos_mask, neg_mask, temperature)
            loss_1 = self.infonce(z2, z1, pos_mask, neg_mask, temperature)
            loss = loss_0*0.5 + loss_1*0.5
        else:  # 划分batch
            node_idxs = list(range(nnodes))
            random.shuffle(node_idxs)
            batches = self.split_batch(node_idxs, self.batch_size)
            loss = 0
            for b in batches:
                weight = len(b) / nnodes
                loss_0 = self.infonce(z1[b], z2[b], pos_mask[:,b][b,:], neg_mask[:,b][b,:], temperature)
                loss_1 = self.infonce(z2[b], z1[b], pos_mask[:,b][b,:], neg_mask[:,b][b,:], temperature)
                loss += (loss_0 + loss_1) / 2.0 * weight
        return loss
    
    def infonce(self, anchor, sample, pos_mask, neg_mask, tau):
        pos_mask = pos_mask.to(anchor)
        neg_mask = neg_mask.to(anchor)
        sim = self.similarity(anchor, sample) / tau
        exp_sim = torch.exp(sim) * neg_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / (pos_mask.sum(dim=1) +  + 1e-10)
        return -loss.mean()


    def similarity(self, h1, h2):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        return h1 @ h2.t()
    
