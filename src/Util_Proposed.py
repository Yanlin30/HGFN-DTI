import csv
import os
import random

import torch
import scipy.sparse as sp

import numpy as np

from loguru import logger
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score, f1_score, log_loss
from lifelines.utils import concordance_index
from scipy.stats import pearsonr
import pickle
from datetime import datetime
import pandas as pd
from tqdm import tqdm
neg_label = 1
pos_label = 0
from torch.utils import data
import sys
sys.path.append("../../DeepPurpose-master")
from DeepPurpose import utils, dataset
def compute_class_metrics(y_label, y_pred, pos_label=1, neg_label=0):
    """
    Compute classification metrics such as sensitivity, specificity, precision, recall, accuracy, and F1 score.

    Args:
        y_label (list or array): True labels.
        y_pred (list or array): Predicted labels.
        pos_label (int, optional): The positive label class. Default is 1.
        neg_label (int, optional): The negative label class. Default is 0.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    # Initialize counts for confusion matrix
    TP = FP = TN = FN = 0

    # Compute confusion matrix values
    for true, pred in zip(y_label, y_pred):
        if pred == pos_label:
            if true == pos_label:
                TP += 1  # True Positive
            else:
                FP += 1  # False Positive
        else:
            if true == neg_label:
                TN += 1  # True Negative
            else:
                FN += 1  # False Negative

    # Log confusion matrix
    logger.info(f"Confusion Matrix: TP={TP}, TN={TN}, FP={FP}, FN={FN}")

    # Compute metrics with checks to avoid division by zero
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 1.0
    specificity = TN / (FP + TN) if (FP + TN) > 0 else 1.0
    recall = sensitivity  # Recall is same as sensitivity
    precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1 = 2 * TP / (2 * TP + FN + FP) if (2 * TP + FN + FP) > 0 else 1.0

    # Return the calculated metrics
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "recall": recall,
        "precision": precision,
        "accuracy": accuracy,
        "f1": f1
    }

def write_csv_withThreshold(file_path, result,threshold_val):
    """Record results to a CSV file."""
    file_exists = pd.io.common.file_exists(file_path)
    timestamp = f'theshold value: {threshold_val} at'+datetime.now().strftime("%m-%d-%H-%M")
    result_df = pd.DataFrame([result])
    result_df['timestamp'] = timestamp
    if file_exists:
        result_df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        result_df.to_csv(file_path, mode='w', header=True, index=False)
def write_csv(file_path, result):
    """Record results to a CSV file."""
    file_exists = pd.io.common.file_exists(file_path)
    timestamp = datetime.now().strftime("%m-%d-%H-%M")
    result_df = pd.DataFrame([result])
    result_df['timestamp'] = timestamp
    if file_exists:
        result_df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        result_df.to_csv(file_path, mode='w', header=True, index=False)

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"create dir: {path}")
    else:
        logger.info(f"dir exists, {path}")

def save_model_old(model, path):
    # Save model state dict and metadata
    torch.save({
        'state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'timestamp': datetime.datetime.now().isoformat()
    }, path)
    logger.info(f"Saved model parameters to {path}")



import os
import torch
import pickle
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

def save_model(model, path, config=None):
    """Save model with complete configuration"""
    save_data = {
        'state_dict': model.state_dict(),
        'config': config if config else {},
        'model_class': model.__class__.__name__,
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__
    }
    torch.save(save_data, path)

def load_model(model,path, strict=True,device=None):
    """Load model with safety checks"""
    data = torch.load(path, map_location=device)


    # Handle DataParallel
    state_dict = data['state_dict']
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k,v in state_dict.items()}

    # Load state dict
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    return model

class DTI_Dataset(data.Dataset):
    # df : a list of data, which includes an index for the pair, an index for entity1 and entity2, from a list that combines all the entities. we want the
    def __init__(self, idx_map, df):
        'Initialization'
        self.idx_map = idx_map
        self.df = df

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        idx1 = self.idx_map[str(self.df.iloc[index].Graph_Drug)]
        idx2 = self.idx_map[self.df.iloc[index].Graph_Target]
        label = self.df.iloc[index].Graph_Label
        drug_encoding = self.df.iloc[index].drug_encoding
        target_encoding = self.df.iloc[index].target_encoding
        v_d = drug_encoding
        v_p = utils.protein_2_embed(target_encoding)
        y = self.df.iloc[index].Seq_Label
        return v_d, v_p , y, idx1, idx2,label

def dti_df_process(df,drug_encoding='MPNN',target_encoding='CNN'):
    df = df.dropna() # drop NaN rows
    seq_drug = df['Drug']
    seq_target = df['Target']
    seq_label = df['Label']
    graph_drug = df['Drug_ID']
    graph_target = df['Target_ID']
    graph_label = df['Label']
    df = pd.DataFrame(zip(seq_drug, seq_target, seq_label,
                          graph_drug, graph_target,graph_label))
    df.rename(columns={0:'Seq_Drug',
                        1: 'Seq_Target',
                        2: 'Seq_Label',
                        3:'Graph_Drug',
                        4:'Graph_Target',
                        5:'Graph_Label'},
                            inplace=True)
    #drug_encoding, target_encoding = 'MPNN', 'CNN' #original
    #drug_encoding, target_encoding = 'MPNN', 'CNN_RNN'
    df = utils.encode_drug(df, drug_encoding, column_name='Seq_Drug')
    df = utils.encode_protein(df, target_encoding, column_name='Seq_Target')
    return df
def df_data_split(df,frac=[0.7, 0.1, 0.2]):
    df = df.sample(frac=1, replace=True, random_state=1) # shuffle

    total = df.shape[0]
    train_idx = int(total*frac[0])
    valid_idx = int(total*(frac[0]+frac[1]))
    df_train = df.iloc[:train_idx]
    df_valid = df.iloc[train_idx:valid_idx]
    df_test = df.iloc[valid_idx:total-1]
    sample_stat(df_train)
    sample_stat(df_valid)
    sample_stat(df_test)
    return df_train, df_valid, df_test
def sample_stat(df):
    neg_samples = df[df.Label == neg_label]
    pos_samples =  df[df.Label == pos_label]
    neg_label_num = neg_samples.shape[0]
    pos_label_num = pos_samples.shape[0]
    logger.info(f'neg/pos:{neg_label_num}/{pos_label_num}, neg:{neg_label_num * 100 //(neg_label_num + pos_label_num)}%, pos:{pos_label_num * 100 //(neg_label_num + pos_label_num)}%')
    return neg_label_num, pos_label_num
def df_data_preprocess(df, oversampling=False, undersampling=True):
    df = df.dropna() # drop NaN rows
    df['Drug_ID'] = df['Drug_ID'].astype(str)
    df = df.rename(columns={"Y": "Label"})
    neg_label_num, pos_label_num = sample_stat(df)
    if oversampling:
        logger.info('oversampling')
        pos_samples = df[df.Label == pos_label]
        for _ in range(1):
           if pd.__version__ >= '2.0.0':
               df = pd.concat([pos_samples, df], ignore_index=True)
           else:
               df = df.append(pos_samples,ignore_index=True)

        # df = get_unobserved_negative_samples(df)
    if undersampling:
        logger.info('undersampling')
        neg_samples = df[df.Label == neg_label][:pos_label_num]
        pos_samples = df[df.Label == pos_label]
        if pd.__version__ >= '2.0.0':
            # Use pd.concat for pandas 2.0 and later
            df = pd.concat([pos_samples, neg_samples], ignore_index=True)
        else:
            # Use append for pandas versions before 2.0
            df = pos_samples.append(neg_samples, ignore_index=True)

    sample_stat(df)
    return df


import torch
import scipy.sparse as sp
import numpy as np

def add_virtual_node(adj: sp.spmatrix):
    adj = adj.tocoo()
    N = adj.shape[0]

    row = np.concatenate([adj.row, np.arange(N), np.full(N, N)])
    col = np.concatenate([adj.col, np.full(N, N), np.arange(N)])
    data = np.concatenate([adj.data, np.ones(N), np.ones(N)])

    new_adj = sp.coo_matrix((data, (row, col)), shape=(N + 1, N + 1))
    return new_adj



def compute_symmetric_normalized_adjacency(A, identity_matrix):
    """
    Computes the symmetric normalized adjacency matrix with self-loops.

    :param A: Sparse adjacency matrix.
    :param identity_matrix: Identity matrix of the same size as A.
    :return: Symmetric normalized adjacency matrix.
    """
    A_tilde = A + 2 * identity_matrix  # Add self-loops
    degrees = np.array(A_tilde.sum(axis=1)).flatten()  # Compute degree matrix
    D_inv_sqrt = sp.diags(np.power(degrees, -0.5, where=degrees > 0))  # Avoid division by zero
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt  # Apply normalization


def generate_sparse_propagation_matrix_old(A, device):
    """
    Generates a sparse propagation matrix in a format suitable for PyTorch.

    :param A: Sparse adjacency matrix.
    :param device: Target device for tensor storage.
    :return: Dictionary containing sparse propagation matrix indices and values.
    """
    I = sp.eye(A.shape[0], format='csr')  # Create identity matrix
    A_tilde_hat = compute_symmetric_normalized_adjacency(A, I).tocoo()  # Normalize and convert to COO format

    indices = np.vstack((A_tilde_hat.row, A_tilde_hat.col))  # Stack row and column indices
    return {
        "indices": torch.LongTensor(indices).to(device),
        "values": torch.FloatTensor(A_tilde_hat.data).to(device),
    }
def generate_sparse_propagation_matrix(A, device):
    """
    Adds a virtual node and generates a sparse propagation matrix for PyTorch.

    :param A: Sparse adjacency matrix (scipy sparse matrix).
    :param device: Target device for tensor storage.
    :return: Dictionary with sparse propagation matrix indices and values.
    """
    A = A.tocoo()  # Ensure COO format
    N = A.shape[0]

    # Step 1: Add virtual node (node index N)
    row = np.concatenate([A.row, np.arange(N), np.full(N, N)])
    col = np.concatenate([A.col, np.full(N, N), np.arange(N)])
    data = np.concatenate([A.data, np.ones(N), np.ones(N)])
    A_virtual = sp.coo_matrix((data, (row, col)), shape=(N + 1, N + 1))

    # Step 2: Add self-loops and compute normalized adjacency
    I = sp.eye(A_virtual.shape[0], format='csr')
    A_tilde_hat = compute_symmetric_normalized_adjacency(A_virtual, I).tocoo()

    # Step 3: Convert to PyTorch sparse format
    indices = np.vstack((A_tilde_hat.row, A_tilde_hat.col))
    return {
        "indices": torch.LongTensor(indices).to(device),
        "values": torch.FloatTensor(A_tilde_hat.data).to(device),
    }

def convert_features_to_sparse_tensor(features, device):
    """
    Converts a dense feature matrix into a sparse PyTorch tensor representation.

    :param features: Dense feature matrix (NumPy or SciPy sparse format).
    :param device: Target device for tensor storage (e.g., 'cpu' or 'cuda').
    :return: Dictionary containing sparse indices, values, and shape of the feature matrix.
    """
    row_indices, col_indices = features.nonzero()  # Extract nonzero indices
    values = np.ones(len(row_indices), dtype=np.float32)  # Assign value 1.0 to all nonzero entries

    sparse_matrix = sp.coo_matrix((values, (row_indices, col_indices)), shape=features.shape, dtype=np.float32)
    indices = np.vstack((sparse_matrix.row, sparse_matrix.col))  # Stack row and column indices

    return {
        "indices": torch.LongTensor(indices).to(device),
        "values": torch.FloatTensor(sparse_matrix.data).to(device),
        "dimensions": sparse_matrix.shape,
    }
