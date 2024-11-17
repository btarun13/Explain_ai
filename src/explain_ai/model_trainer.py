
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


# computes a node embedding using GINConv layers, then uses pooling to predict graph level properties
class GINGraphPropertyModel(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout_p):
      super(GINGraphPropertyModel, self).__init__()
      # fields used for computing node embedding
      self.node_encoder = AtomEncoder(hidden_dim)

      self.convs = torch.nn.ModuleList(
          [torch_geometric.nn.conv.GINConv(MLP([hidden_dim, hidden_dim, hidden_dim])) for idx in range(0, num_layers)]
      )
      self.bns = torch.nn.ModuleList(
          [torch.nn.BatchNorm1d(num_features = hidden_dim) for idx in range(0, num_layers - 1)]
      )
      self.dropout_p = dropout_p
      # end fields used for computing node embedding
      # fields for graph embedding
      self.pool = global_add_pool
      self.linear_hidden = torch.nn.Linear(hidden_dim, hidden_dim)
      self.linear_out = torch.nn.Linear(hidden_dim, output_dim)
      # end fields for graph embedding
    def reset_parameters(self):
      for conv in self.convs:
        conv.reset_parameters()
      for bn in self.bns:
        bn.reset_parameters()
      self.linear_hidden.reset_parameters()
      self.linear_out.reset_parameters()
    def forward(self, batched_data):
      x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
      # compute node embedding
      x = self.node_encoder(x)
      for idx in range(0, len(self.convs)):
        x = self.convs[idx](x, edge_index)
        if idx < len(self.convs) - 1:
          x = self.bns[idx](x)
          x = torch.nn.functional.relu(x)
          x = torch.nn.functional.dropout(x, self.dropout_p, training=self.training)
      # note x is raw logits, NOT softmax'd
      # end computation of node embedding
      # convert node embedding to a graph level embedding using pooling
      x = self.pool(x, batch)
      x = torch.nn.functional.dropout(x, self.dropout_p, training=self.training)
      # transform the graph embedding to the output dimension
      # MLP after graph embed ensures we are not requiring the raw pooled node embeddings to be linearly separable
      x = self.linear_hidden(x)
      x = torch.nn.functional.relu(x)
      x = torch.nn.functional.dropout(x, self.dropout_p, training=self.training)
      out = self.linear_out(x)
      return out
    
    # can be used with multiple task outputs (like for molpcba) or single task output;
# and supports using just the first output of a multi-task model if applied to a single task (for pretraining molpcba and transferring to molhiv)
def train(model, device, data_loader, optimizer, loss_fn):
  model.train()
  for step, batch in enumerate(tqdm(data_loader, desc="Training batch")):
    batch = batch.to(device)
    if batch.x.shape[0] != 1 and batch.batch[-1] != 0:
      # ignore nan targets (unlabeled) when computing training loss.
      non_nan = batch.y == batch.y
      loss = None
      optimizer.zero_grad()
      out = model(batch)
      non_nan = non_nan[:min(non_nan.shape[0], out.shape[0])]
      batch_y = batch.y[:out.shape[0], :]
      # for crudely adapting multitask models to single task data
      if batch.y.shape[1] == 1:
        out = out[:, 0]
        batch_y = batch_y[:, 0]
        non_nan = batch_y == batch_y
        loss = loss_fn(out[non_nan].reshape(-1, 1)*1., batch_y[non_nan].reshape(-1, 1)*1.)
      else:
        loss = loss_fn(out[non_nan], batch_y[non_nan])
      loss.backward()
      optimizer.step()
  return loss.item()

def eval(model, device, loader, evaluator, save_model_results=False, save_filename=None):
  model.eval()
  y_true = []
  y_pred = []
  for step, batch in enumerate(tqdm(loader, desc="Evaluation batch")):
      batch = batch.to(device)
      if batch.x.shape[0] == 1:
          pass
      else:
          with torch.no_grad():
              pred = model(batch)
              # for crudely adapting multitask models to single task data
              if batch.y.shape[1] == 1:
                pred = pred[:, 0]
              batch_y = batch.y[:min(pred.shape[0], batch.y.shape[0])]
              y_true.append(batch_y.view(pred.shape).detach().cpu())
              y_pred.append(pred.detach().cpu())
  y_true = torch.cat(y_true, dim=0).numpy()
  y_pred = torch.cat(y_pred, dim=0).numpy()
  input_dict = {"y_true": y_true.reshape(-1, 1) if batch.y.shape[1] == 1 else y_true, "y_pred": y_pred.reshape(-1, 1) if batch.y.shape[1] == 1 else y_pred}
  if save_model_results:
      single_task = len(y_true.shape) == 1 or y_true.shape[1] == 1
      if single_task:
          data = {
              'y_pred': y_pred.squeeze(),
              'y_true': y_true.squeeze()
          }
          pd.DataFrame(data=data).to_csv('ogbg_graph_' + save_filename + '.csv', sep=',', index=False)
      else:
          num_tasks = y_true.shape[1]
          for task_idx in range(num_tasks):
              data = {
                  'y_pred': y_pred[:, task_idx].squeeze(),
                  'y_true': y_true[:, task_idx].squeeze()
              }
              pd.DataFrame(data=data).to_csv('ogbg_graph_' + save_filename + f'_task_{task_idx}.csv', sep=',', index=False)
  return evaluator.eval(input_dict)

def combined_eval(gin_model, gcn_model, device, loader, evaluator):
    gin_model.eval()
    gcn_model.eval()

    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Evaluation batch")):
      batch = batch.to(device)
      if batch.x.shape[0] == 1:
        pass
      else:
        with torch.no_grad():
          pred1 = gin_model(batch)
          pred2 = gcn_model(batch)
          pred = (pred1 + pred2) / 2

          if batch.y.shape[1] == 1:
            pred = pred[:, 0]
          batch_y = batch.y[:min(pred.shape[0], batch.y.shape[0])]
          y_true.append(batch_y.view(pred.shape).detach().cpu())
          y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true.reshape(-1, 1) if batch.y.shape[1] == 1 else y_true, "y_pred": y_pred.reshape(-1, 1) if batch.y.shape[1] == 1 else y_pred}
    return evaluator.eval(input_dict)


# Run combined evaluation on the validation set
# combined_val_performance = combined_eval(gin_model, gcn_model, device, valid_loader, evaluator)
# print("Combined Validation Performance:", combined_val_performance)




# config = {
#  'device': 'cuda',
#  'dataset_id': 'ogbg-molhiv',
#  'num_layers': 2,
#  'hidden_dim': 64,
#  'dropout': 0.5,
#  'learning_rate': 0.001,
#  'epochs': 25,
#  'batch_size': 32,
#  'weight_decay': 1e-6
# }



def build_model_gin(config, train_loader, val_loader,evaluator, device):
    
    num_tasks = 1
    model = GINGraphPropertyModel(config['hidden_dim'], num_tasks, config['num_layers'], config['dropout']).to(device)
    print(f"parameter count: {sum(p.numel() for p in model.parameters())}")
    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_model = None
    best_valid_metric_at_save_checkpoint = 0
    best_train_metric_at_save_checkpoint = 0

    for epoch in range(1, 1 + config["epochs"]):
        if epoch == 10:
        # reduce learning rate at this point
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']*0.5, weight_decay=config['weight_decay'])
    loss = train(model, device, train_loader, optimizer, loss_fn)
    train_perf = eval(model, device, train_loader, evaluator)
    val_perf = eval(model, device, val_loader, evaluator)
  # if not args.hide_test_metric:
  #     # not necessary as output unused during train loop but needed for reproduciblility as affects number of random number generations, affecting ability to generate previously observed outputs depending on seed
  #     test_perf = eval(model, device, test_loader, evaluator)
    eval_metric = 'rocauc' 
    train_metric, valid_metric = train_perf[eval_metric], val_perf[eval_metric]
    if valid_metric >= best_valid_metric_at_save_checkpoint and train_metric >= best_train_metric_at_save_checkpoint:
        print(f"New best validation score: {valid_metric} ({eval_metric}) without training score regression")
        best_valid_metric_at_save_checkpoint = valid_metric
        best_train_metric_at_save_checkpoint = train_metric
        best_model = copy.deepcopy(model)
    
    print(f'Dataset {config["dataset_id"]}, ',
        f'Epoch: {epoch}, ',
        f'Train: {train_metric:.6f} ({eval_metric}), ',
        f'Valid: {valid_metric:.6f} ({eval_metric}), ')

    with open(f"best_{config['dataset_id']}_gin_model_{config['num_layers']}_layers_{config['hidden_dim']}_hidden.pkl", "wb") as f:
       pickle.dump(best_model, f)

    train_metric = eval(best_model, device, train_loader, evaluator)[eval_metric]
    valid_metric = eval(best_model, device, val_loader, evaluator, save_model_results=True, save_filename=f"gin_{config['dataset_id']}_valid")[eval_metric]
    test_metric = None

    print(f'Best model for {config["dataset_id"]} (eval metric {eval_metric}): ',
        f'Train: {train_metric:.6f}, ',
        f'Valid: {valid_metric:.6f} ')
    print(f"parameter count: {sum(p.numel() for p in best_model.parameters())}")

    return best_model