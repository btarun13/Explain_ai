

import captum
import os
import argparse
import pickle
import torch
import torch_geometric
from torch_geometric.nn import GINConv, global_add_pool,GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import MLP
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
import pandas as pd
import copy
import numpy as np
import random
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
from ogb.utils.mol import smiles2graph
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset



class MoleculeDataset(InMemoryDataset):
    def __init__(self, data_list, transform=None):
        # Initialize the dataset with InMemoryDataset superclass
        super().__init__(None, transform)
        
        # Collate the data list into data and slices (PyTorch Geometric expects this format)
        self.data, self.slices = self.collate(data_list)
    
    @property
    def processed_file_names(self):
        # Since we're storing data in memory, we can just return an empty list
        return []

    def process(self):
        # No need to process files, as we are already providing data directly
        pass




# column name should be 'smiles' and target column should be 'target' 
def get_data_formate_from_smiles(smile_df):
    data_list = []

    for i,j in enumerate(list(smile_df['smiles'])):
        graph_data_sample = smiles2graph(j)
        # Convert edge_index to a tensor and ensure it's in the correct shape
        edge_index = torch.tensor(graph_data_sample['edge_index'], dtype=torch.long)
        # Convert edge_feat and node_feat to tensors
        edge_attr = torch.tensor(graph_data_sample['edge_feat'], dtype=torch.long)
        x = torch.tensor(graph_data_sample['node_feat'], dtype=torch.long)
        # Convert the target column from the DataFrame to a tensor
        y = torch.tensor(hiv_data['target'][i], dtype=torch.long).reshape(1,1)
        num_nodes = torch.tensor(graph_data_sample['num_nodes'], dtype=torch.long)

        data = Data(edge_index=edge_index, edge_attr=edge_attr, x=x, y=y, num_nodes=num_nodes)
        data_list.append(data)

    return data_list

def generate_splits(data_list,split_ratio): ### split_ratio should be between 0 and 1
   sample_size = int( float(split_ratio) * len(data_list)) 
   
   train_sample = data_list[:sample_size]
   valid_sample = data_list[sample_size:]

def get_graph_property_loader(train_sample,valid_sample, batch_size):
    batch_size = 32
    train_dataset = MoleculeDataset(train_sample)
    val_dataset = MoleculeDataset(valid_sample)

    evaluator = Evaluator(name='rocauc')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    return train_loader, val_loader, evaluator