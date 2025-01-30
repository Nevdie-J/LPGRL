import os
import torch
import torch as th
import numpy as np
import random
import torch.nn as nn
from torch_geometric.utils import to_dense_adj, remove_self_loops, negative_sampling
from torch.utils.data import DataLoader, TensorDataset
from torch_sparse import SparseTensor
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from munkres import Munkres
import pandas as pd

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def normalize_adj(edge_index, edge_weight=None, mode='sym'):
    if edge_weight == None:
        edge_weight = torch.ones(edge_index.size(1)).to(edge_index)
    
    num_nodes = edge_index.max().item() + 1
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight, sparse_sizes=(num_nodes, num_nodes))
    if mode == "sym":
        inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1)) + 1e-10)
        return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
    elif mode == "row":
        inv_degree = 1. / (adj.sum(dim=1) + 1e-10)
        return inv_degree[:, None] * adj
    elif mode == "None":
        return adj

class LogReg(nn.Module):
    def __init__(self, hid_dim, n_classes):
        super(LogReg, self).__init__()

        self.fc = nn.Linear(hid_dim, n_classes)

    def forward(self, x):
        res = self.fc(x)
        return res

def eval_node_classification(h, labels, n_classes, train_mask, val_mask, test_mask):
    train_embs = h[train_mask]
    val_embs = h[val_mask]
    test_embs = h[test_mask]

    train_labels = labels[train_mask]
    val_labels = labels[val_mask]
    test_labels = labels[test_mask]
    
    def train():
        clf.train()
        optimizer.zero_grad()
        logits = clf(train_embs)
        preds = torch.argmax(logits, dim=1)
        train_acc = torch.sum(preds == train_labels).float() / train_labels.shape[0]
        loss = loss_fn(logits, train_labels)
        loss.backward()
        optimizer.step()
        
        return train_acc
    
    def test():
        clf.eval()
        with th.no_grad():
            val_logits = clf(val_embs)
            test_logits = clf(test_embs)

            val_preds = th.argmax(val_logits, dim=1)
            test_preds = th.argmax(test_logits, dim=1)

            val_acc = th.sum(val_preds == val_labels).float() / val_labels.shape[0]
            test_acc = th.sum(test_preds == test_labels).float() / test_labels.shape[0]
        return val_acc, test_acc
    
    best_val_acc = 0
    eval_acc = 0

    clf = LogReg(h.shape[1], n_classes).to(h.device)
    optimizer = th.optim.Adam(clf.parameters(), lr=0.01, weight_decay=0.0)

    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1500):
        train()
        val_acc, test_acc = test()
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            if test_acc > eval_acc:
                eval_acc = test_acc

    return eval_acc.cpu().data, best_val_acc


def eval_link_prediction(model, h1, h2, pos_edge_index, neg_edge_index, batch_size=2**16):
    def batch_eval(model, h1, h2, edge_index, batch_size):
        preds = []
        for perm in DataLoader(range(edge_index.size(1)), batch_size):
            edge = edge_index[:, perm]
            preds.append(model(h1, h2, edge).squeeze().detach().cpu().sigmoid())
        pred = torch.cat(preds, dim=0)
        return pred
    
    pos_pred = batch_eval(model, h1, h2, pos_edge_index, batch_size)
    neg_pred = batch_eval(model, h1, h2, neg_edge_index, batch_size)
    
    pred = torch.cat([pos_pred, neg_pred], dim=0)
    pos_y = pos_pred.new_ones(pos_pred.shape[0])
    neg_y = neg_pred.new_zeros(neg_pred.shape[0])
    
    y = torch.cat([pos_y, neg_y], dim=0)
    y, pred = y.cpu().numpy(), pred.cpu().numpy()
    auc, ap = roc_auc_score(y, pred), average_precision_score(y, pred)
    return auc, ap


def eval_link_prediction_baseline(h, train_data, test_data):
    def batch_eval(model, h, edge_index, batch_size):
        preds = []
        for perm in DataLoader(range(edge_index.size(1)), batch_size):
            edge = edge_index[:, perm]
            preds.append(model(h, edge).squeeze().detach().cpu().sigmoid())
        pred = torch.cat(preds, dim=0)
        return pred
    
    h_ = StandardScaler().fit_transform(h.cpu().numpy())
    h = torch.tensor(h_).to(h)
    
    link_decoder = LinkDecoder(h.shape[1], h.shape[1]*2).to(h)
    optimizer = th.optim.Adam(link_decoder.parameters(), lr=0.01, weight_decay=0.0)
    for i in range(1000):
        link_decoder.train()
        optimizer.zero_grad()
        
        pos = link_decoder(h, train_data.pos_edge_label_index)
        neg = link_decoder(h, train_data.neg_edge_label_index)
        loss = link_decoder.loss(pos, neg)
        loss.backward()
        optimizer.step()

    link_decoder.eval()
    pos_pred = batch_eval(link_decoder, h, test_data.pos_edge_label_index, batch_size=2**16)
    neg_pred = batch_eval(link_decoder, h, test_data.neg_edge_label_index, batch_size=2**16)
    
    pred = torch.cat([pos_pred, neg_pred], dim=0)
    pos_y = pos_pred.new_ones(pos_pred.shape[0])
    neg_y = neg_pred.new_zeros(neg_pred.shape[0])
    
    y = torch.cat([pos_y, neg_y], dim=0)
    y, pred = y.cpu().numpy(), pred.cpu().numpy()
    
    return roc_auc_score(y, pred), average_precision_score(y, pred)


class LinkDecoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim=1, dropout=0.3):
        super().__init__()
        self.decoder_mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim)
           ,nn.ReLU()
           ,nn.Linear(hid_dim, out_dim)
        )
    
    def forward(self, h, edge_index):
        u, v = edge_index[0], edge_index[1]
        # out = self.decoder_mlp(torch.cat([h[u], h[v]], dim=-1))
        out = self.decoder_mlp(h[u] * h[v])
        return out
    
    def decoder_in(self, h, edge_index, sigmoid=False):
        u, v = edge_index[0], edge_index[1]
        val = (h[u] * h[v]).sum(dim=1)
        return torch.sigmoid(val) if sigmoid else val

    def loss(self, pos, neg):
        pos_loss = -torch.log(pos.sigmoid() + 1e-15).mean()
        neg_loss = -torch.log(1 - neg.sigmoid() + 1e-15).mean()

        return pos_loss + neg_loss

