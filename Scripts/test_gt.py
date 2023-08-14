"""
"""

import math
import time

from torch_geometric.data import Data
import torch_geometric
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import ZINC, GNNBenchmarkDataset
from ogb.graphproppred import PygGraphPropPredDataset

from Misc.config import config 
from Misc.utils import edge_tensor_to_list
from Misc.cyclic_adjacency_transform import CyclicAdjacencyTransform

colors_feat = ['white', 'red', 'orange', 'yellow', 'blue', 'green', 'grey', 'pink']
colors_type = ['red', 'orange', 'yellow', 'blue', 'green', 'grey', 'pink']
cololrs_edges = ['orange', 'red', 'yellow', 'blue', 'green', 'grey', 'pink', 'purple']
default_color = 'white'
default_color_edge = 'black'

def visualize(data, name, colors, v_feat_dim = None, e_feat_dim = None):
    G = nx.DiGraph(directed=True)

    edge_list = edge_tensor_to_list(data.edge_index)

    for i, edge in enumerate(edge_list):
        G.add_edge(edge[0], edge[1], color=default_color_edge if (e_feat_dim is None) else cololrs_edges[data.edge_attr[i, e_feat_dim]])

    color_map_vertices = [default_color for _ in G]
    color_map_edges = [default_color_edge for _ in range(data.edge_index.shape[1])]

    if v_feat_dim is not None:
        node_types = data.x[:, v_feat_dim].tolist()
        for i, node in enumerate(G):
            color_map_vertices[i] = colors[node_types[node]]

    edges = G.edges()
    color_map_edges = [G[u][v]['color'] for u,v in edges]

    nx.draw_kamada_kawai(G, node_color=color_map_vertices,  edge_color=color_map_edges, with_labels=True, font_weight='bold', arrows=True)
    plt.savefig(f'Imgs/{name}.png')
    plt.close()

def main():
    # Graph 1
    edge_index1 = torch.tensor([[0, 1, 1, 2, 0, 2, 2, 3, 3, 4, 4, 7, 7, 6, 3, 6, 3, 7, 5, 7],
                               [1, 0, 2, 1, 2, 0, 3, 2, 4, 3, 7, 4, 6, 7, 6, 3, 7, 3, 7, 5]], dtype=torch.long)
    x1 = torch.tensor([[0],[1],[2],[3], [4], [0], [1], [2]])
    edge_attr1 = torch.tensor([[0], [0],[1], [1],[2], [2],[3], [3], [4], [4], [0], [0], [1], [1], [2], [2], [3], [3], [4], [4]])
    
    # Graph 2
    edge_index2 = torch.tensor([[0, 1, 1, 2, 2, 0, 2, 3, 3, 4, 4, 5, 5, 3],
                               [1, 0, 2, 1, 0, 2, 3, 2, 4, 3, 5, 4, 3, 5]], dtype=torch.long)   
    x2 = torch.tensor([[0],[1],[2],[3], [4], [0]])
    edge_attr2 = torch.tensor([[0], [0],[1], [1],[2], [2],[3], [3], [4], [4], [0], [0], [1], [1]])
    
    # Graph 3
    edge_index3 = torch.tensor([[0, 1, 1, 2, 2, 0, 2, 3, 3, 4, 4, 2],
                               [1, 0, 2, 1, 0, 2, 3, 2, 4, 3, 2, 4]], dtype=torch.long)   
    x3 = torch.tensor([[0],[1],[2],[3], [4]])
    edge_attr3 = torch.tensor([[0], [0],[1], [1],[2], [2],[3], [3], [4], [4], [0], [0]])
    
    # Graph 4
    edge_index4 = torch.tensor([[0, 1, 1, 2, 2, 0, 2, 3, 3, 4, 4, 5, 5, 6, 6, 2, 2, 5],
                                [1, 0, 2, 1, 0, 2, 3, 2, 4, 3, 5, 4, 6, 5, 2, 6, 5, 2]], dtype=torch.long)
    x4 = torch.tensor([[0],[1],[2],[3], [4], [0], [1]])
    edge_attr4 = torch.tensor([[0], [0],[1], [1],[2], [2],[3], [3], [4], [4], [0], [0], [1], [1], [2], [2], [3], [3]])
    
    # Graph 5
    edge_index5 = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
                                [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]], dtype=torch.long)    
    x5 = torch.tensor([[0],[1],[2],[3], [4], [0]])
    edge_attr5 = torch.tensor([[0], [0],[1], [1],[2], [2],[3], [3], [4], [4], [0], [0]])
    
    
    # Graph 6
    edge_index6 = torch.tensor([[0, 1, 1, 2, 2, 0, 2, 3, 3, 4, 4, 5, 4, 6, 6, 7, 7, 8, 8, 6],
                                [1, 0, 2, 1, 0, 2, 3, 2, 4, 3, 5, 4, 6, 4, 7, 6, 8, 7, 6, 8]], dtype=torch.long)
    x6 = torch.tensor([[0],[1],[2],[3], [4], [0], [1], [2], [3]])
    edge_attr6 = torch.tensor([[0], [0],[1], [1],[2], [2],[3], [3], [4], [4], [0], [0], [1], [1], [2], [2], [3], [3], [4], [4]])

    # Graph 7
    edge_index7 = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 1, 3, 1, 4],
                                [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5, 3, 1, 4, 1]], dtype=torch.long)    
    x7 = torch.tensor([[0],[1],[2],[3], [4], [0]])
    edge_attr7 = torch.tensor([[0], [0],[1], [1],[2], [2],[3], [3], [4], [4], [0], [0], [0], [0],[1], [1]])

    edge_indices = [edge_index1, edge_index2, edge_index3, edge_index4, edge_index5, edge_index6]
    edge_indices = [edge_index1, edge_index2, edge_index3, edge_index4, edge_index5, edge_index6, edge_index7]
    xs = [x1, x2, x3, x4, x5, x6, x7]
    edge_attrs = [edge_attr1, edge_attr2, edge_attr3, edge_attr4, edge_attr5, edge_attr6, edge_attr7]


    for i in range(len(edge_indices)):
        edge_index = edge_indices[i]
        data =  Data(edge_index=edge_index)

        x = xs[i]
        edge_attr = edge_attrs[i]
        data =  Data(edge_index=edge_index, x=x, edge_attr=edge_attr)
        
        # print(f"Before trafo: {data}")
        data.x = data.x + 1
        data.edge_attr = data.edge_attr + 1
        visualize(data, f"graph_{i}", colors_feat, v_feat_dim=0, e_feat_dim=0)
        data.x = data.x -1
        data.edge_attr = data.edge_attr - 1
        transform = CyclicAdjacencyTransform(debug=True)
        
        transformed_data = transform(data)

        # print(f"After trafo: {transformed_data}")

        visualize(transformed_data, f"graph_{i}_transformed_types", colors_type, v_feat_dim=0, e_feat_dim=0)
        visualize(transformed_data, f"graph_{i}_transformed_og_feats", colors_feat, v_feat_dim=1, e_feat_dim=2)
        visualize(transformed_data, f"graph_{i}_transformed_distance", colors_feat, v_feat_dim=1, e_feat_dim=1)

    # quit()
    print("Preparing dataset")
    # ds = ZINC(root=config.DATA_PATH, subset=True, split="train")
    ds = PygGraphPropPredDataset(root=config.DATA_PATH, name="ogbg-molhiv")
    
    print("Running on dataset")
    transform = CyclicAdjacencyTransform(debug=True)
    start = time.time()
    for i, data in enumerate(ds):        
        # data.x = data.x + 1
        # data.edge_attr = data.edge_attr + 1
        # visualize(data, f"zinc_{i}", colors_feat)
        # data.x = data.x -1
        # data.edge_attr = data.edge_attr - 1
        
        
        transformed_data = transform(data)

        # print(f"After trafo: {transformed_data}")

        # visualize(transformed_data, f"zinc_{i}_transformed_types", colors_type, v_feat_dim=0, e_feat_dim=0)
        # visualize(transformed_data, f"zinc_{i}_transformed_distance", colors_feat, v_feat_dim=0, e_feat_dim=1)
        
        # print(f"\r{i}", end="")
        
        # if (i + 1) % 11 == 0:
        #     break

    end = time.time()
    print(f"\rRuntime  {end - start:.2f}s")
    
if __name__ == "__main__":
    main()