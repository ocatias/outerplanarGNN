import os
import csv

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC, GNNBenchmarkDataset, GNNBenchmarkDataset, LRGBDataset
import torch.optim as optim
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import ToUndirected, Compose, OneHotDegree
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

from Models.gnn import GNN
from Models.encoder import NodeEncoder, EdgeEncoder, ZincAtomEncoder, EgoEncoder
from Models.mlp import MLP
from Models.ESAN.transform import policy2transform
from Models.ESAN.conv import ZINCGINConv, GINConv
from Models.ESAN.models import DSSnetwork
from Misc.drop_features import DropFeatures
from Misc.add_zero_edge_attr import AddZeroEdgeAttr
from Misc.pad_node_attr import PadNodeAttr
from Misc.cyclic_adjacency_transform import CyclicAdjacencyTransform
from Misc.scheduler import get_cosine_schedule_with_warmup

def get_max_hamiltonian_cycle_length(dataset):
    if dataset == "peptides-func":
        return -1
    elif dataset == "ogbg-molhiv":
        return 44
    elif dataset == "ZINC":
        return 30
    return 44

def get_transform(args, split = None):
    transforms = []
    if args.dataset.lower() == "csl":
        transforms.append(OneHotDegree(5))
        
    # Pad features if necessary (needs to be done after adding additional features from other transformation)
    if args.dataset.lower() == "csl":
        transforms.append(AddZeroEdgeAttr(args.emb_dim))
        transforms.append(PadNodeAttr(args.emb_dim))
      
    if args.do_drop_feat:
        transforms.append(DropFeatures(args.emb_dim))
        
    if args.use_cat:
        transforms.append(CyclicAdjacencyTransform(spiderweb=args.use_spiderweb))

    if args.model == "DSS":
        print("Importing 3-egonets policy")
        transforms.append(policy2transform("ego_nets", 3))

    return Compose(transforms)

def load_dataset(args, config):
    transform = get_transform(args)

    if transform is None:
        dir = os.path.join(config.DATA_PATH, args.dataset, "Original")
    else:
        print(repr(transform))
        trafo_str = repr(transform).replace("\n", "")
        dir = os.path.join(config.DATA_PATH, args.dataset, trafo_str)

    if args.dataset.lower() in ["zinc", "zinc100k"]:
        use_subset = args.dataset.lower() == "zinc"
        args.dataset = "ZINC"
        datasets = [ZINC(root=dir, subset=use_subset, split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    elif args.dataset.lower() == "cifar10":
        datasets = [GNNBenchmarkDataset(name ="CIFAR10", root=dir, split=split, pre_transform=Compose([ToUndirected(), transform])) for split in ["train", "val", "test"]]
    elif args.dataset.lower() == "cluster":
        datasets = [GNNBenchmarkDataset(name ="CLUSTER", root=dir, split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    elif args.dataset.lower() in ["ogbg-molhiv", "ogbg-ppa", "ogbg-code2", "ogbg-molpcba", "ogbg-moltox21", "ogbg-molesol", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltoxcast", "ogbg-molfreesolv", "ogbg-mollipo"]:
        dataset = PygGraphPropPredDataset(root=dir, name=args.dataset.lower(), pre_transform=transform)
        split_idx = dataset.get_idx_split()
        datasets = [dataset[split_idx["train"]], dataset[split_idx["valid"]], dataset[split_idx["test"]]]
    elif args.dataset.lower() == "csl":
        all_idx = {}
        for section in ['train', 'val', 'test']:
            with open(os.path.join(config.SPLITS_PATH, "CSL",  f"{section}.index"), 'r') as f:
                reader = csv.reader(f)
                all_idx[section] = [list(map(int, idx)) for idx in reader]
        dataset = GNNBenchmarkDataset(name ="CSL", root=dir, pre_transform=transform)
        datasets = [dataset[all_idx["train"][args.split]], dataset[all_idx["val"][args.split]], dataset[all_idx["test"][args.split]]]
    elif args.dataset.lower() in ["exp", "cexp"]:
        dataset = PlanarSATPairsDataset(name=args.dataset, root=dir, pre_transform=transform)
        split_dict = dataset.separate_data(args.seed, args.split)
        datasets = [split_dict["train"], split_dict["valid"], split_dict["test"]]
    elif args.dataset.lower() == "peptides-func":
        datasets = [LRGBDataset(root=dir, name='Peptides-func', split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    elif args.dataset.lower() == "peptides-struct":
        datasets = [LRGBDataset(root=dir, name='Peptides-struct', split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    elif args.dataset.lower() == "pascalvoc-sp":
        datasets = [LRGBDataset(root=dir, name='PascalVOC-SP', split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    elif args.dataset.lower() == "coco-sp":
        datasets = [LRGBDataset(root=dir, name='COCO-SP', split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    elif args.dataset.lower() == "pcqm-contact":
        datasets = [LRGBDataset(root=dir, name='PCQM-Contact', split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    else:
        raise NotImplementedError("Unknown dataset")
     
    if args.model == "DSS":
        print("Using ESAN")
        train_loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True, follow_batch=['subgraph_idx'])
        val_loader = DataLoader(datasets[1], batch_size=args.batch_size, shuffle=False, follow_batch=['subgraph_idx'])
        test_loader = DataLoader(datasets[2], batch_size=args.batch_size, shuffle=False, follow_batch=['subgraph_idx'])
    else:
        train_loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(datasets[1], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(datasets[2], batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def add_1_to_all_in_ls(ls):
    return list(map(lambda x: x+1, ls))

def get_model(args, num_classes, num_vertex_features, num_tasks):
    node_feature_dims = []
    edge_feature_dims = []
    model = args.model.lower()
    
    cat_add = 1 if args.use_cat else 0
    
    if args.activation.lower() == "relu":
        activation = torch.nn.ReLU()
    elif args.activation.lower() == "gelu":
        activation = torch.nn.GELU()
    
    if args.use_cat:
        node_feature_dims.append(7)
        edge_feature_dims += [9, get_max_hamiltonian_cycle_length(args.dataset)]
        
    if args.dataset.lower() == "zinc"and not args.do_drop_feat:
        edge_feature_dims += [4 + cat_add]
        node_feature_dims.append(28 + cat_add)
        node_encoder = NodeEncoder(emb_dim=args.emb_dim, feature_dims=node_feature_dims)
        edge_encoder =  EdgeEncoder(emb_dim=args.emb_dim, activation=activation, feature_dims=edge_feature_dims)
    elif args.dataset.lower() in ["peptides-func", "ogbg-molhiv", "ogbg-molpcba", "ogbg-moltox21", "ogbg-molesol", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltoxcast", "ogbg-molfreesolv", "ogbg-mollipo"] and not args.do_drop_feat:
        edge_feature_dims += add_1_to_all_in_ls(get_bond_feature_dims())
        node_feature_dims += add_1_to_all_in_ls(get_atom_feature_dims())
        print("node_feature_dims: ", node_feature_dims)
        node_encoder, edge_encoder = NodeEncoder(args.emb_dim, feature_dims=node_feature_dims), EdgeEncoder(args.emb_dim, activation=activation, feature_dims=edge_feature_dims)
    else:
        node_encoder, edge_encoder = lambda x: x, lambda x: x
            
    if model in ["gin", "gcn", "gat"]:  
        return GNN(num_classes, num_tasks, activation, args.num_layers, args.emb_dim, 
                gnn_type = model, virtual_node = args.use_virtual_node, drop_ratio = args.drop_out, JK = args.JK, 
                graph_pooling = args.pooling, edge_encoder=edge_encoder, node_encoder=node_encoder, 
                use_node_encoder = args.use_node_encoder, num_mlp_layers = args.num_mlp_layers, between_repr_factor = args.between_repr_factor, residual = args.use_residual_conection)
    elif args.model.lower() == "mlp":
            return MLP(num_features=num_vertex_features, num_layers=args.num_layers, hidden=args.emb_dim, 
                    num_classes=num_classes, num_tasks=num_tasks, dropout_rate=args.drop_out, graph_pooling=args.pooling)
    elif args.model == "DSS":    
        if "zinc" in args.dataset.lower():
            GNNConv = ZINCGINConv
        else:
            GNNConv = GINConv
    
        model = DSSnetwork(num_layers=args.num_layers, in_dim=args.emb_dim , emb_dim=args.emb_dim, num_tasks=num_tasks*num_classes,
                        feature_encoder=node_encoder, GNNConv=GNNConv, bond_encoder=edge_encoder)
            
    else: # Probably don't need other models
        raise ValueError("Unknown model name")

    return model


def get_optimizer_scheduler(model, args, finetune = False):
    
    if finetune:
        lr = args.lr2
    else:
        lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    args.lr_scheduler_decay_steps,
                                                    gamma=args.lr_scheduler_decay_rate)
    elif args.lr_scheduler == 'None':
        scheduler = None
    elif args.lr_scheduler == "ReduceLROnPlateau":
         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                mode='min',
                                                                factor=args.lr_scheduler_decay_rate,
                                                                patience=args.lr_schedule_patience,
                                                                verbose=True)
    elif args.lr_scheduler == "Cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = args.linear_warmup, num_training_steps = args.epochs)
    
    else:
        raise NotImplementedError(f'Scheduler {args.lr_scheduler} is not currently supported.')

    print(f"optimizer: {optimizer}")
    print(f"scheduler: {scheduler}")
    return optimizer, scheduler

def get_loss(args):
    metric_method = None
    if args.dataset.lower() == "zinc":
        loss = torch.nn.L1Loss()
        metric = "mae"
    elif args.dataset.lower() in ["ogbg-molesol", "ogbg-molfreesolv", "ogbg-mollipo"]:
        loss = torch.nn.L1Loss()
        metric = "rmse (ogb)"
        metric_method = get_evaluator(args.dataset)
    elif args.dataset.lower() in ["cifar10", "csl", "exp", "cexp"]:
        loss = torch.nn.CrossEntropyLoss()
        metric = "accuracy"
    elif args.dataset in ["ogbg-molhiv", "ogbg-moltox21", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molsider", "ogbg-moltoxcast"]:
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "rocauc (ogb)" 
        metric_method = get_evaluator(args.dataset)
    elif args.dataset == "ogbg-ppa":
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "accuracy (ogb)" 
        metric_method = get_evaluator(args.dataset)
    elif args.dataset in ["ogbg-molpcba", "ogbg-molmuv"]:
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "ap (ogb)" 
        metric_method = get_evaluator(args.dataset)
    elif args.dataset.lower() == "peptides-func":
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "ap" 
    else:
        raise NotImplementedError("No loss for this dataset")
    
    return {"loss": loss, "metric": metric, "metric_method": metric_method}

def get_evaluator(dataset):
    evaluator = Evaluator(dataset)
    eval_method = lambda y_true, y_pred: evaluator.eval({"y_true": y_true, "y_pred": y_pred})
    return eval_method