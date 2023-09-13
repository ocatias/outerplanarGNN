"""
 _._     _,-'""`-._
(,-.`._,'(       |\`-/|
    `-.-' \ )-`( , o o)
          `-    \`_`"'-
            CAT.
"""

import io
import time
import subprocess
import json
from collections import defaultdict, ChainMap

from Misc.graph_edit import *
from Misc.utils import list_of_lists_to_list

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import coalesce, sort_edge_index

#
# Constants
#
label_ham_cycle = 0
label_articulation_vertex = 1
label_pooling_vertex = 2
label_block_vertex = 3
label_original_Vertex = 4
label_global_pool = 5

pos_edge_type = 0
# Hamiltonian cycle distance = 0 -> not in a hamiltonian cycle
pos_ham_dis = 1

label_edge_original = 0
label_edge_ham_cycle = 1
label_edge_ham_pool = 2
label_edge_pool_block = 3
label_edge_pool_art = 4

# Edge from articulation node to original node or articulation node
label_edge_art_original = 5
label_edge_block_ham = 6
label_edge_pool_ham = 7
label_edge_shortcut = 8

#
# CODE
#

def maybe_add_edge_attr(has_edge_attr, edge_attr, e_shape, label, nr_edges = 1):
    if has_edge_attr:
        new_feat = torch.zeros([nr_edges, e_shape])
        new_feat[:, pos_edge_type] = label
        return torch.cat((edge_attr, new_feat), dim=0)

def get_hamiltonian_cycles(g: Data, config): 
    from Misc.run_converter import to_aids_format

    tic = time.time()
    graphstring = ''
    
    # graph conversion to textual input format for all graphs in the current batch
    # assumes undirected graphs and ignores labels
    file = io.BytesIO()
    to_aids_format(g, 0, file, undirected=True)
    graphstring = file.getvalue().decode('latin1')
    
    # the actual computation of outerplanarity and Hamiltonian cycles in a subprocess
    cmd = [config['binfile'], '-sw', '-']
    proc = subprocess.run(args=cmd, capture_output=True, input=graphstring.encode("utf-8"))

    # parsing of the results (directly from stdout of the process)
    jstr = proc.stdout.decode("utf-8")
    # print(jstr)
    jsobjects = json.loads(jstr)

    return jsobjects

def edge_in_ham_cycles(edge, list_vertices_in_ham_cycle):
    return int(edge[0]) in list_vertices_in_ham_cycle and int(edge[1]) in list_vertices_in_ham_cycle

def get_ham_cycle(edge, ham_cycle_dict, ham_cycles):
    """
    Returns id of the hamiltonian cycle of an edge
    """
    cycles1 = ham_cycle_dict[int(edge[0])]
    cycles2 = ham_cycle_dict[int(edge[1])]
    
    # Intersection between the two list of cycles
    cycles = list(set(cycles1) & set(cycles2))
    
    # An edge can be in at most one hamiltonian cycle
    assert len(cycles) <= 1
    
    return cycles[0] if (len(cycles) == 1) else None
    
class CyclicAdjacencyTransform(BaseTransform):
    """
            
    """
    def __init__(self, debug = False):
        self.config = {'binfile': './outerplanaritytest', 'verbose': True}
        self.debug = debug
        print("Created CyclicAdjacencyTransform")

    def __call__(self, data: Data):
        edge_index, x, edge_attr = data.edge_index, data.x, data.edge_attr 
        has_edge_attr = edge_attr is not None 
        has_vertex_feat = x is not None
        
        if (len(edge_attr.shape) == 1):
            edge_attr = torch.unsqueeze(edge_attr, 1)

        nr_vertices_in_og_graph = get_nr_vertices(data)

        # Increment features by 1 to make room for new "empty" edges / vertices
        # This assumes that all features are categorical
        if has_vertex_feat:
            x += 1
            x_shape = x.shape[1]

        # Shift features (see: shifting of vertex features)
        if has_edge_attr:
            edge_attr += 1
            e_shape = edge_attr.shape[1]+2

            # Make space for labels and hamiltonian cycle distance
            edge_attr = torch.cat((torch.zeros([edge_attr.shape[0], 2]), edge_attr), dim = 1)


        ham_cycle_info = get_hamiltonian_cycles(data, self.config)[0]
        # ham_cycles = ham_cycle_info[0]["hamiltonianCycles"]
        print(ham_cycle_info)
        
        
        
        ls_ham_cycles_dict = list(map(lambda x: x["hamiltonianCycles"], ham_cycle_info['ccs']))
        print(f"ham_cycles_dict: {ls_ham_cycles_dict}")
        
        # Different hamiltonian cycles in different connected components can have the same id: make id unique
        temp_dict = {}
        latest_key = 0
        for dictionary in ls_ham_cycles_dict:
            dictionary = dict(sorted(dictionary.items(), key=lambda item: int(item[0]), reverse=True))
            print(f"dictionary: {dictionary}")
            for (key, value) in dictionary.items():
                key = int(key)
                if key >= latest_key:
                    latest_key -= 1
                    key = latest_key 
                temp_dict[key] = value
        ham_cycles_dict = temp_dict
        print(f"temp_dict: {temp_dict}")
        
        blocks = dict(ChainMap(*list(map(lambda x: x["blocks"], ham_cycle_info['ccs']))))

        print(f"blocks: {len(blocks)}, ls_ham_cycles_dict: {len(ham_cycles_dict)}, ccs: {len(ham_cycle_info['ccs'])}")
        # if len(blocks) != len(ham_cycles_dict) or len(ham_cycle_info['ccs']) > 1:
        #     print("quitting")
        #     if has_vertex_feat:
        #         data.x = x.type(data.x.type())
        #     if has_edge_attr:
        #         data.edge_attr = edge_attr.type(data.edge_attr.type())
        #     return data
        
        print(f"ham_cycles_dict: {ham_cycles_dict}")

        ham_cycles = list(ham_cycles_dict.values())
        print(f"ham_cycles: {ham_cycles}")
        
        shortcut_edges = list(map(lambda x: x["shortcutEdges"], ham_cycle_info['ccs']))
        shortcut_edges = list_of_lists_to_list(shortcut_edges)
        print(f"shortcut_edges: {shortcut_edges}")
        
        articulation_vertices = []
        
        # Dict to map vertices in ham cycle to id of cycle
        ham_cycle_dict = defaultdict(lambda: [])
        
        ham_cycle_id_to_raw_id = {}
        raw_id_to_block_vertex = {}


        ham_cycles_dict = dict(ChainMap(*ls_ham_cycles_dict))
         
        # for i, ls in enumerate(ham_cycles):
        #     for vertex in ls:
        #         ham_cycle_dict[(vertex)].append(int(i))

        for i, (key, ls) in enumerate(ham_cycles_dict.items()):
            ham_cycle_id_to_raw_id[i] = key
            for vertex in ls:
                ham_cycle_dict[(vertex)].append(i)
              
        print(f"ham_cycle_id_to_raw_id: {ham_cycle_id_to_raw_id}")  
        vertices_in_ham_cycles = list(set(ham_cycle_dict.keys()))
        vertices_in_ham_cycles.sort()       
        new_edge_index = torch.clone(edge_index)
        nr_vertices_in_og_graph = get_nr_vertices(data)
        
        # Duplicate and orient hamiltonian cycles
        # also collect articulation vertices

        # (Vertex in og graph, cycle idx) -> vertex in new graph
        vertex_to_original_dict = {}
        vertex_to_duplicate_dict = {}

        created_vertices = 0
        already_seen_vertices = []
        if has_vertex_feat:
            labels = [label_original_Vertex for _ in range(nr_vertices_in_og_graph)]
        for cycle_idx, cycle in enumerate(ham_cycles):
            for vertex in cycle:
                # Create original vertex entry
                if vertex in already_seen_vertices:
                    idx = nr_vertices_in_og_graph + created_vertices
                    created_vertices += 1
                    
                    if has_vertex_feat:
                        x = torch.cat((x, torch.unsqueeze(x[vertex], 0)), dim=0)
                        labels.append(label_ham_cycle)
                else:
                    idx = vertex
                    already_seen_vertices.append(vertex)
                    if has_vertex_feat:
                        labels[idx] = label_ham_cycle
                vertex_to_original_dict[(vertex, cycle_idx)] = idx

                # Duplicate
                idx = nr_vertices_in_og_graph + created_vertices
                created_vertices += 1
                vertex_to_duplicate_dict[(vertex, cycle_idx)] = idx
                if has_vertex_feat:
                    x = torch.cat((x, torch.unsqueeze(x[vertex], 0)), dim=0)
                    labels.append(label_ham_cycle)

        if has_vertex_feat:
            x = torch.cat((torch.unsqueeze(torch.tensor(labels), 1), x), dim= 1)
            
        edges_outside_ham_cycle = []
        get_duplicate_vertex = lambda vertex, cycle: vertex_to_duplicate_dict[(vertex, cycle)]

        for i in range(edge_index.shape[1]):
            ham_cycle_id = get_ham_cycle(edge_index[:, i], ham_cycle_dict, ham_cycles)
            edge = (int(edge_index[0, i]), int(edge_index[1, i]))
            
            # If edge is part of hamiltionian cycle: clone / orient it
            if ham_cycle_id is not None:   
                if has_edge_attr:
                    edge_attr[i, pos_edge_type] = label_edge_ham_cycle    
                    
                
                ham_cycle = ham_cycles[ham_cycle_id]
                pos_in_ham_cycle1, pos_in_ham_cycle2 = ham_cycle.index(int(edge[0])), ham_cycle.index(int(edge[1]))
                modulus = len(ham_cycle) 

                # This checks if the vertices are part of the hamiltonian cycle in a clockwise / counter clockwise fashion
                # Clockwise (+1): move edge if some of the vertices are not original
                if (pos_in_ham_cycle1 + 1) % modulus == pos_in_ham_cycle2:
                    for j in [0, 1]:
                        vertex_idx = vertex_to_original_dict.get((edge[j], ham_cycle_id))
                        if vertex_idx is not edge[j]:
                            new_edge_index[j, i] = vertex_idx
                            
                    if has_edge_attr:
                        edge_attr[i, pos_ham_dis] = 1
                
                # Counter clockwise (-1): create new vertices / edges 
                elif (pos_in_ham_cycle1 + modulus - 1) % modulus == pos_in_ham_cycle2:                    
                    new_edge_index[0, i] = get_duplicate_vertex(int(new_edge_index[0, i]), ham_cycle_id)
                    new_edge_index[1, i] = get_duplicate_vertex(int(new_edge_index[1, i]), ham_cycle_id)
                    
                    if has_edge_attr:
                        edge_attr[i, pos_ham_dis] = 1

                # Neither (these are edges that are not part of the hamiltonian cycle): duplicate
                else:
                    v1 = get_duplicate_vertex(int(new_edge_index[0, i]), ham_cycle_id)
                    v2 = get_duplicate_vertex(int(new_edge_index[1, i]), ham_cycle_id)
                    new_edge_index = add_dir_edge(new_edge_index, v1, v2)
                    
                    if has_edge_attr:
                        new_feat = torch.zeros([1, e_shape])
                        new_feat[0, pos_edge_type] = label_edge_original
                        
                        # Add distance in hamiltonian cycle to new and old edge
                        distance_1_to_2 = (pos_in_ham_cycle2 + len(ham_cycle) - pos_in_ham_cycle1) % len(ham_cycle)
                        distance_2_to_1 = (pos_in_ham_cycle1 + len(ham_cycle) - pos_in_ham_cycle2) % len(ham_cycle)
                        new_feat[0, pos_ham_dis] = distance_2_to_1
                        edge_attr[i, pos_ham_dis] = distance_1_to_2
                        
                        edge_attr = torch.cat((edge_attr, new_feat), dim=0)
                        
                for j in [0, 1]:
                    if len(ham_cycle_dict[int(edge[j])]) > 1:
                        articulation_vertices += [int(edge[j])]
            else:
                # Check if vertices incident to edge are articulation vertices
                articulation_vertices += [int(edge[j]) for j in [0, 1] if (int(edge[j]) in vertices_in_ham_cycles)]
                edges_outside_ham_cycle += [i]
            
        nr_vertices_in_graph_with_duplication = nr_vertices_in_og_graph + created_vertices
        articulation_vertices = list(set(articulation_vertices))

        # Create vertices that pool the representation for a single vertex
        created_vertices = 0
        vertex_to_pooling = {}
        for cycle_idx, cycle in enumerate(ham_cycles):
            for vertex in cycle:
                if vertex in articulation_vertices:
                    idx = nr_vertices_in_graph_with_duplication + created_vertices
                    created_vertices += 1
                    
                    vertex_to_pooling[(vertex, cycle_idx)] = idx
                    vertex_og = vertex_to_original_dict[(vertex, cycle_idx)]
                    vertex_duplicate = vertex_to_duplicate_dict[(vertex, cycle_idx)]
                   
                    new_edge_index = add_undir_edge(new_edge_index, vertex_og, idx)             
                    new_edge_index = add_undir_edge(new_edge_index, vertex_duplicate, idx)
                    
                    edge_attr = maybe_add_edge_attr(has_edge_attr, edge_attr, e_shape, label_edge_ham_pool, 4)

                    if has_vertex_feat:
                        new_feat = torch.cat((torch.tensor([label_pooling_vertex]), x[vertex, 1:]))
                        x = torch.cat((x, torch.unsqueeze(new_feat, 0)), dim=0)
                        
        nr_vertices_in_graph_with_pooling = nr_vertices_in_graph_with_duplication + created_vertices  
        
        # Create block vertices
        created_vertices = 0
        cycle_to_block_vertex = {}
        for cycle_idx, cycle in enumerate(ham_cycles):
            idx = nr_vertices_in_graph_with_pooling + created_vertices
            created_vertices += 1
            cycle_to_block_vertex[cycle_idx] = idx
            raw_id_to_block_vertex[ham_cycle_id_to_raw_id[cycle_idx]] = idx
            if has_vertex_feat:
                new_feat = torch.cat((torch.tensor([label_block_vertex]), torch.zeros(x_shape)))
                x = torch.cat((x, torch.unsqueeze(new_feat, 0)), dim=0)
            
            for vertex in cycle:
                new_edge_index = add_undir_edge(new_edge_index, vertex_to_original_dict[(vertex, cycle_idx)], idx)
                new_edge_index = add_undir_edge(new_edge_index, vertex_to_duplicate_dict[(vertex, cycle_idx)], idx)
                edge_attr = maybe_add_edge_attr(has_edge_attr, edge_attr, e_shape, label_edge_block_ham, 4)
                
                if vertex in articulation_vertices:
                    pooling_vertex = vertex_to_pooling[(int(vertex), cycle_idx)]
                    new_edge_index = add_undir_edge(new_edge_index, pooling_vertex, idx)
                    edge_attr = maybe_add_edge_attr(has_edge_attr, edge_attr, e_shape, label_edge_pool_block, 2)
                    
        print(f"cycle_to_block_vertex: {cycle_to_block_vertex}")

        nr_vertices_in_graph_with_block = nr_vertices_in_graph_with_pooling + created_vertices
        
        # Create articulation vertices
        created_vertices = 0
        vertex_to_articulation = {}
        for i, articulation_vertex in enumerate(articulation_vertices):
            idx = nr_vertices_in_graph_with_block + i
            vertex_to_articulation[articulation_vertex] = idx
            created_vertices += 1
            if has_vertex_feat:
                new_feat = torch.cat((torch.tensor([label_articulation_vertex]), x[articulation_vertex, 1:]))
                x = torch.cat((x, torch.unsqueeze(new_feat, 0)), dim=0)

            for cycle_idx, cycle in enumerate(ham_cycles):
                if articulation_vertex in cycle:
                    # Edge from articulation vertex to pooling vertex
                    pooling_vertex = vertex_to_pooling[(int(articulation_vertex), cycle_idx)]
                    new_edge_index = add_undir_edge(new_edge_index, pooling_vertex, idx)
                    edge_attr = maybe_add_edge_attr(has_edge_attr, edge_attr, e_shape, label_edge_pool_art, 2)
                    
        nr_vertices_in_new_graph = nr_vertices_in_graph_with_block + created_vertices
        
        # Add a virtual node pooling blocks:
        idx = nr_vertices_in_new_graph
        if has_vertex_feat:
            new_feat = torch.cat((torch.tensor([label_global_pool]), torch.zeros(x_shape)))
            x = torch.cat((x, torch.unsqueeze(new_feat, 0)), dim=0)
                
        for block in cycle_to_block_vertex.values():
            new_edge_index = add_undir_edge(new_edge_index, block, idx) 
            edge_attr = maybe_add_edge_attr(has_edge_attr, edge_attr, e_shape, label_edge_pool_ham, 2)
        nr_vertices_in_new_graph += 1
        
        # Add shortcut edges
        print(f"raw_id_to_block_vertex: {raw_id_to_block_vertex}")
        for edge in shortcut_edges:
            print(edge)
            p, q = edge[0], edge[1]
            print(f"p: {p}, q: {q}")
            # If incident vertices are block vertices then map them to the newly created block vertices
            if p < 0 and str(p) in raw_id_to_block_vertex:
                p = raw_id_to_block_vertex[str(p)]
            if q < 0 and str(q) in raw_id_to_block_vertex:
                q = raw_id_to_block_vertex[str(q)]
                
            if p < 0 or q < 0:
                continue
                
            print(f"p: {p}, q: {q}")
            new_edge_index = add_undir_edge(new_edge_index, p, q) 
            edge_attr = maybe_add_edge_attr(has_edge_attr, edge_attr, e_shape, label_edge_shortcut, 2)
        
        
        # Clean up edges: move edges from articulation vertices in the hamiltonian cycles to the articulation representation vertices
        for i in edges_outside_ham_cycle:
            for j in [0, 1]:
                if new_edge_index[j, i] in articulation_vertices:
                    new_edge_index[j, i] = vertex_to_articulation[int(new_edge_index[j, i])]

        data.edge_index = new_edge_index.type(data.edge_index.type())
        if has_vertex_feat:
            data.x = x.type(data.x.type())
        if has_edge_attr:
            data.edge_attr = edge_attr.type(data.edge_attr.type())
        data.num_nodes = x.shape[0]


        assert data.edge_index.shape[1] == data.edge_attr.shape[0]
        if data.edge_index.shape[1]> 0:
            assert data.x.shape[0] >= (torch.max(data.edge_index) + 1)
            assert torch.min(data.edge_index) >= 0
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

"""
END of CAT
"""