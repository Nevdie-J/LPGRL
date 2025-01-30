import argparse

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric.utils import remove_self_loops, degree, dropout_adj
import seaborn as sns
from torch_sparse import SparseTensor, spspmm
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
from alive_progress import alive_bar
from torch_geometric.utils.num_nodes import maybe_num_nodes
import wandb
import os

from data_utils import *
from common_utils import *
from plot_utils import *
from models import *


def main(args, seed=0):
    
    print(args)
    setup_seed(seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    data = load_data(name=args.dataset, missing_edge_rate=0.0)
    data = data.to(device)
    
    if 'link_prediction' in args.task:
        train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True, split_labels=True, add_negative_train_samples=True)(data)
        data = train_data
        data.edge_index, _ = dropout_adj(data.edge_index, p=args.missing_edge_rate, force_undirected=True, num_nodes=data.num_nodes)
    
    model = LPGRL(in_dim=data.x.shape[1]
                , hid_dim=args.hidden_channel
                , out_dim=args.embedding_size
                , n_layers=args.num_layers
                , batch_size=args.cl_batch_size
                , n_nodes=data.x.shape[0]
                , n_feats=data.x.shape[1]
                , use_bn_g=args.use_bn_g
                , use_bn_m=args.use_bn_m
                , adj_dropout=args.adj_dropout
                , p=args.p
                , use_intr=args.use_intr).to(device)
    model.set_mask_knn(x=data.x, edge_index=data.edge_index, k=args.k, name=args.dataset)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_loss = float("inf")
    cnt_wait = 0
    best_epoch = 0

    tag = str(int(time.time()))
    
    # 迭代训练
    with alive_bar(args.epochs) as bar:
        for epoch in range(args.epochs):
            x, edge_index = data.x, data.edge_index
            # edge_index1, edge_index2 = data.edge_index, data.edge_index
            loss = model(x, edge_index, args.alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch: {epoch}, loss: {loss:0.4f}")

            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                cnt_wait = 0
                th.save(model.state_dict(), './model-pkl/LPGRL/best_model_'+ args.dataset + tag + '.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print("Early stopping")
                break
            bar()


    # downstream task
    print('Loading {}th epoch'.format(best_epoch + 1))
    model.load_state_dict(th.load('./model-pkl/LPGRL/best_model_'+ args.dataset + tag + '.pkl'))
    model.eval()
    node_embedding, h_g, p, h_m, projg, projm, projp = model.get_embedding(data.x, data.edge_index)

    
    print("=== Evaluation ===")
    test_res, val_res = [], []
    for run in range(args.runs):
        seed = run + 42
        
        setup_seed(seed)

        if 'node_classification' in args.task:
            cur = 0 if (data.train_mask.shape[1]==1) else (run % data.train_mask.shape[1])
            train_mask, val_mask, test_mask = data.train_mask[:, cur], data.val_mask[:, cur], data.test_mask[:, cur]

            eval_test, eval_val = eval_node_classification(node_embedding, data.y, data.n_classes
                                    , train_mask, val_mask, test_mask)
        
        if 'link_prediction' in args.task:
            eval_test, _ = eval_link_prediction(model.adj_decoder, h_g, p, test_data.pos_edge_label_index, test_data.neg_edge_label_index, batch_size=2**16)
            # eval_test, _ = eval_link_prediction_baseline(node_embedding, data, test_data)
            eval_val = torch.tensor([0.0])
        
        print(run, f'{args.task} accuracy:{eval_test:.4f}')
        test_res.append(eval_test)
        val_res.append(eval_val)
    
    val_results = [v.item() for v in val_res]
    val_mean = np.mean(val_results, axis=0) * 100
    val_values = np.asarray(val_results, dtype=object)
    val_uncertainty = np.max(
        np.abs(sns.utils.ci(sns.algorithms.bootstrap(val_values, func=np.mean, n_boot=1000), 95) - val_values.mean()))
    
    test_results = [v.item() for v in test_res]
    test_mean = np.mean(test_results, axis=0) * 100
    test_values = np.asarray(test_results, dtype=object)
    test_uncertainty = np.max(
        np.abs(sns.utils.ci(sns.algorithms.bootstrap(test_values, func=np.mean, n_boot=1000), 95) - test_values.mean()))
    print(f'LPGRL on {args.dataset}】test acc mean = {test_mean:.2f} ± {test_uncertainty * 100:.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ESSENTIAL
    parser.add_argument('-dataset', type=str, default='cora')
    parser.add_argument('-gpu_id', type=int, default=0)
    parser.add_argument('-runs', type=int, default=10)
    parser.add_argument('-missing_edge_rate', type=float, default=0.0)
    parser.add_argument('-epochs', type=int, default=800)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-wd', type=float, default=1e-5)
    parser.add_argument("-patience", type=int,default=200)
    parser.add_argument('-task', nargs='+', type=str, default=['node_classification'])
    
    # Model Hyper-param
    parser.add_argument('-embedding_size', type=int, default=256)
    parser.add_argument('-hidden_channel', type=int, default=256)
    parser.add_argument('-num_layers', type=int, default=2)
    parser.add_argument('-num_proj_layers', type=int, default=1)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-cl_batch_size', type=int, default=0)
    parser.add_argument('-k', type=int, default=30)
    parser.add_argument('-use_bn_g', type=bool, default=False)
    parser.add_argument('-use_bn_m', type=bool, default=False)
    parser.add_argument('-alpha', type=float, default=0.5)
    parser.add_argument('-adj_dropout', type=float, default=0.3)
    parser.add_argument('-p', type=float, default=0.7)
    parser.add_argument('-use_intr', type=bool, default=False)
    
    
    args = parser.parse_args()


    main(args)