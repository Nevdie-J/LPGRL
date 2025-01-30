import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.datasets import WikipediaNetwork, Planetoid, Amazon,\
    Coauthor, WebKB, AttributedGraphDataset, HeterophilousGraphDataset, Actor, WikiCS, Twitch
import torch_geometric.transforms as T
from torch_geometric.utils import dropout_adj, index_to_mask\
, remove_self_loops, to_undirected, degree, remove_isolated_nodes, add_self_loops
from ogb.nodeproppred import PygNodePropPredDataset
from torch_sparse import SparseTensor

def load_data(root='/workspace/data', name='cora', is_random_split=False, missing_edge_rate=0.0):

    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=root + '/Planetoid', name=name
                            , transform=T.NormalizeFeatures()
                            )
    elif name in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(root=root + '/WebKB', name = name
                        , transform=T.NormalizeFeatures()
                        )
    elif name in ['actor']:
        dataset = Actor(root=root + '/Actor')
    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=root + '/WikipediaNetwork', name=name
                                   , transform=T.NormalizeFeatures()
                                   , geom_gcn_preprocess=True
                                   )
    
    elif name in ['crocodile']:
        dataset = WikipediaNetwork(root=root + '/WikipediaNetwork', name=name
                                   , transform=T.NormalizeFeatures(), geom_gcn_preprocess=False)
    elif name in ['computers', 'photo']:
        dataset = Amazon(root=root + '/Amazon', name=name, transform=T.NormalizeFeatures())
    elif name in ['cs', 'physics']:
        dataset = Coauthor(root=root + '/Coauthor', name=name, transform=T.NormalizeFeatures())
    elif name in ['wiki-cs']:
        dataset = WikiCS(root=root + '/Wiki-CS')
    elif name in ['blogcatalog', 'flickr']:
        dataset = AttributedGraphDataset(root + '/AttributedGraphDataset', name)
    elif name in ['amazon-ratings', 'minesweeper', 'questions', 'roman-empire']:
        dataset = HeterophilousGraphDataset(root + '/hetero_data', name
                                            # , transform=T.NormalizeFeatures()
                                            )
    elif name in ['twitch-de']:
        dataset = Twitch(root + '/Twitch', 'DE')
    elif name in ['twitch-en']:
        dataset = Twitch(root + '/Twitch', 'EN')
    elif name in ['syn-0.2']:
        dataset = CSBM(n=3000, d=5, ratio=0.2, p=3500, mu=1)
    elif name in ['syn-0.9']:
        dataset = CSBM(n=3000, d=5, ratio=0.9, p=3500, mu=1)
        
    if name in ['squirrel-filtered', 'chameleon-filtered']:
        if name == 'squirrel-filtered':
            path = '/workspace/data/WikipediaNetwork/squirrel/raw/squirrel_filtered.npz'
        elif name == 'chameleon-filtered':
            path = '/workspace/data/WikipediaNetwork/chameleon/raw/chameleon_filtered.npz'
        data = np.load(path)
        x, y, edge_index = torch.tensor(data['node_features']), torch.tensor(data['node_labels']), torch.tensor(data['edges']).T
        train_mask, test_mask, val_mask = torch.tensor(data['train_masks']), torch.tensor(data['test_masks']), torch.tensor(data['val_masks'])
        data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask, test_mask=test_mask, val_mask=val_mask)
    else:   
        data = dataset[0]    
    if name in ['flickr']:
        data.x = data.x.to_dense()
        data.num_nodes = data.x.shape[0]
    
    
    if missing_edge_rate > 0.0:
        print(f'Original num_edges: {data.num_edges}')
        data.edge_index, _ = dropout_adj(data.edge_index, p=missing_edge_rate, force_undirected=True, num_nodes=data.num_nodes)
        print(f'after missing num_edges: {data.num_edges}')
    
    data.edge_index = remove_self_loops(data.edge_index)[0]
    data.edge_index = to_undirected(data.edge_index)
    data.edge_sparse = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value=torch.ones(data.edge_index.size(1)), sparse_sizes=(data.x.shape[0], data.x.shape[0]))
    data.n_classes = len(torch.unique(data.y))
    
    # 数据集划分
    if name in ['computers', 'photo', 'cs', 'physics', 'wiki-cs', 'crocodile'
                , 'squirrel', 'chameleon', 'blogcatalog', 'flickr', 'squirrel-filtered', 'chameleon-filtered'
                , 'twitch-de', 'twitch-en'
                ] or is_random_split == True:
        train_mask, val_mask, test_mask = get_split(data.x.shape[0])
    else:
        train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
        print(f'Using public split of {name}! 20 per class/30 per class/1000 for train/val/test.')
    if len(train_mask.shape) < 2:
        train_mask = train_mask.unsqueeze(1)
    if len(val_mask.shape) < 2:
        val_mask = val_mask.unsqueeze(1)
    if len(test_mask.shape) < 2:
        test_mask = test_mask.unsqueeze(1)
    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
    
    # data.edge_index_2_hop = generate_two_hop_edges(data.edge_index)
    # data, _ = remove_isolated(data)
    # data.degrees = degree(edge_index_[0], num_nodes=data.x.shape[0])
    # data.degree_centrality = data.degrees / (data.x.shape[0] - 1)

    # print(f'Edge homophily: {edge_homo_ratio(data)}')

    return data

def edge_homo_ratio(data):
    sum = 0
    for i in range(len(data.edge_index[0])):
        if data.y[data.edge_index[0][i]] == data.y[data.edge_index[1][i]]:
            sum += 1
    return sum / len(data.edge_index[0])

def get_split(num_samples, train_ratio = 0.1, test_ratio = 0.8, num_splits = 10):
    
    # 进行十次随机的划分
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)

    trains, vals, tests = [], [], []

    for _ in range(num_splits):
        indices = torch.randperm(num_samples)

        train_mask = torch.zeros(num_samples, dtype=torch.bool)
        train_mask.fill_(False)
        train_mask[indices[:train_size]] = True

        test_mask = torch.zeros(num_samples, dtype=torch.bool)
        test_mask.fill_(False)
        test_mask[indices[train_size: test_size + train_size]] = True

        val_mask = torch.zeros(num_samples, dtype=torch.bool)
        val_mask.fill_(False)
        val_mask[indices[test_size + train_size:]] = True

        trains.append(train_mask.unsqueeze(1))
        vals.append(val_mask.unsqueeze(1))
        tests.append(test_mask.unsqueeze(1))

    train_mask_all = torch.cat(trains, 1)
    val_mask_all = torch.cat(vals, 1)
    test_mask_all = torch.cat(tests, 1)

    return train_mask_all, val_mask_all, test_mask_all

def remove_isolated(data):
    n_nodes_before =data.x.shape[0]
    _, _, mask = remove_isolated_nodes(data.edge_index, num_nodes=n_nodes_before)
    data = data.subgraph(mask)
    n_nodes_after =data.x.shape[0]
    nodes_removed = n_nodes_before - n_nodes_after
    return(data, nodes_removed)


def rand_train_test_idx(label, train_prop, valid_prop, test_prop, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    labeled_nodes = torch.where(label != -1)[0]

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)
    test_num = int(n * test_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:train_num + valid_num + test_num]

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}


def CSBM(n, d, ratio, p, mu, train_prop=.6, valid_prop=.2, test_prop=0.2):
    Lambda = np.sqrt(d) * (2 * ratio - 1)
    c_in = d + np.sqrt(d) * Lambda
    c_out = d - np.sqrt(d)*Lambda
    print('c_in: ', c_in, 'c_out: ', c_out)
    y = np.ones(n)
    y[int(n/2)+1:] = -1
    y = np.asarray(y, dtype=int)

    # creating edge_index
    edge_index = [[], []]
    for i in range(n-1):
        for j in range(i+1, n):
            if y[i]*y[j] > 0:
                Flip = np.random.binomial(1, c_in/n)
            else:
                Flip = np.random.binomial(1, c_out/n)
            if Flip > 0.5:
                edge_index[0].append(i)
                edge_index[1].append(j)
                edge_index[0].append(j)
                edge_index[1].append(i)

    # creating node features
    x = np.zeros([n, p])
    u = np.random.normal(0, 1/np.sqrt(p), [1, p])
    for i in range(n):
        Z = np.random.normal(0, 1, [1, p])
        x[i] = np.sqrt(mu/n)*y[i]*u + Z/np.sqrt(p)

    data = Data(x=torch.tensor(x, dtype=torch.float32),
                edge_index=torch.tensor(edge_index),
                y=torch.tensor((y + 1) // 2, dtype=torch.int64))

    data.coalesce()

    splits_lst = rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
    data.train_mask = index_to_mask(splits_lst['train'], data.num_nodes)
    data.val_mask = index_to_mask(splits_lst['valid'], data.num_nodes)
    data.test_mask = index_to_mask(splits_lst['test'], data.num_nodes)
    return [data]