import copy
import csv
import os

import scipy.sparse as sp
from ogb.graphproppred import PygGraphPropPredDataset
import torch
from torch_geometric.datasets import ZINC
from torch_geometric.utils import get_laplacian, to_dense_adj, to_scipy_sparse_matrix
import torch_geometric.transforms as T
from torch.linalg import eig
from tqdm import tqdm

from Misc.config import config
from Misc.cyclic_adjacency_transform import CyclicAdjacencyTransform
import numpy as np
from scipy.linalg import null_space, solve_continuous_lyapunov

CAT = CyclicAdjacencyTransform(debug=False, spiderweb=False)
spiderCAT = CyclicAdjacencyTransform(debug=False, spiderweb=True)
node_remover = T.Compose([T.remove_isolated_nodes.RemoveIsolatedNodes()])


def symmetric_resistance(graph):
    laplacian = get_laplacian(graph.edge_index)
    dense_laplacian = to_dense_adj(edge_index=laplacian[0], edge_attr=laplacian[1])[0]
    eigenvalues = eig(dense_laplacian).eigenvalues
    assert eigenvalues.imag.sum() == 0.0
    eigenvalues = eigenvalues.real.sort()[0][1:]  # Take real part, sort, exclude smallest eigenvalue.
    resistance = graph.num_nodes * (1 / eigenvalues).sum()  # R = n * tr(L+)

    return resistance, resistance / graph.num_nodes


def directed_resistance(graph):
    # From https://arxiv.org/pdf/1310.5163.pdf
    if graph.num_nodes < 2:
        # Resistance of single nodes taken as zero.
        return np.array([0])

    laplacian = get_laplacian(graph.edge_index)
    dense_laplacian = to_dense_adj(edge_index=laplacian[0], edge_attr=laplacian[1])[0]
    n = dense_laplacian.shape[0]

    # Create subspace perpendicular to vector 1n.
    A = np.zeros((n, n))
    A[0] = np.ones(n)
    Q = null_space(A).T  # Ortonormal basis of nullspace, with basis as row vectors.

    # Projection matrix onto that subspace. -> np.matmul(Q, np.ones((n, 1))) should be ~0.
    # pi = np.identity(n) - (1/float(n)) * (np.matmul(np.ones((n, 1)), np.ones((n, 1)).T))

    L = Q.dot(dense_laplacian).dot(Q.T)  # Reduced laplacian.
    S = solve_continuous_lyapunov(L.astype(np.float64),
                                  np.identity(n - 1).astype(np.float64))  # Solution of Lyapunov equation.
    X = 2 * Q.T.dot(S).dot(Q)

    # Get resistances.
    R = np.zeros((n, n))
    for k, row in enumerate(X):
        for j, cell in enumerate(row):
            R[k][j] = X[k][k] + X[j][j] - 2 * X[k][j]

    return R


def compute_dataset_resistance(dataset, transform_func, resistance_func):
    mean_resistance = []
    max_resistance = []
    sum_resistance = []

    for i, graph in enumerate(tqdm(dataset)):

        graph = transform_func(graph)
        graph = node_remover(graph)

        # Check for multiple components.
        adj = to_scipy_sparse_matrix(graph.edge_index, num_nodes=graph.num_nodes)
        num_components, component = sp.csgraph.connected_components(adj, connection="weak")
        _, count = np.unique(component, return_counts=True)

        for i in range(0, num_components):
            subset = np.in1d(component, count.argsort()[i])
            graph_component = graph.clone().subgraph(torch.from_numpy(subset).to(torch.bool))
            R = resistance_func(graph_component)
            mean_resistance.append(R.mean())
            max_resistance.append(R.max())
            sum_resistance.append(R.sum())

    return mean_resistance, sum_resistance, max_resistance


ogbg_datasets = ["ogbg-molbace", "ogbg-moltox21", "ogbg-molesol", "ogbg-molbbbp",
                 "ogbg-molsider", "ogbg-moltoxcast", "ogbg-mollipo", "ogbg-molhiv"]

datasets = ["zinc"] + ogbg_datasets

splits = ["train", "test"]

dataset_name = "ogbg-molhiv"
split = "train"

if dataset_name == "zinc":
    ds = ZINC(root=config.DATA_PATH, subset=True, split=split)
    spider_ds = copy.deepcopy(ds)
elif dataset_name in ogbg_datasets:
    ds = PygGraphPropPredDataset(root=config.DATA_PATH, name=dataset_name)
    split_idx = ds.get_idx_split()
    ds = ds[split_idx[split]]
    spider_ds = copy.deepcopy(ds)
else:
    raise "Sure?"
    ds = ZINC(root=config.DATA_PATH, subset=True, split=split)
    spider_ds = copy.deepcopy(ds)


results_path = "Results/"
digits = 3

r_mean, r_sum, r_max = compute_dataset_resistance(ds, CAT, directed_resistance)
sr_mean, sr_sum, sr_max = compute_dataset_resistance(spider_ds, spiderCAT, directed_resistance)

header = ["method",
          "avg_mean_res", "std_mean_res",
          "avg_tot_res", "std_tot_res",
          "avg_max_res", "std_max_res"]
average_cat = ["CAT",
               np.round(np.mean(r_mean), digits), np.round(np.std(r_mean), digits),
               np.round(np.mean(r_sum), digits), np.round(np.std(r_sum), digits),
               np.round(np.mean(r_max), digits), np.round(np.std(r_max), digits)]
average_spider_cat = ["spiderCAT",
                      np.round(np.mean(sr_mean), digits), np.round(np.std(sr_mean), digits),
                      np.round(np.mean(sr_sum), digits), np.round(np.std(sr_sum), digits),
                      np.round(np.mean(sr_max), digits), np.round(np.std(sr_max), digits)]

with open(os.path.join(results_path, dataset_name+split+".csv"), 'w') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(header)
    csvwriter.writerow(average_cat)
    csvwriter.writerow(average_spider_cat)

with open(os.path.join(results_path, dataset_name+split+"_all.csv"), 'w') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(["r_mean"] + r_mean)
    csvwriter.writerow(["r_sum"] + r_sum)
    csvwriter.writerow(["r_max"] + r_max)
    csvwriter.writerow(["sr_mean"] + sr_mean)
    csvwriter.writerow(["sr_sum"] + sr_sum)
    csvwriter.writerow(["sr_max"] + sr_max)