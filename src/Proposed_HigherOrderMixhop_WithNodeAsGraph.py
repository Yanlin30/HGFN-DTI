import time, sys
from datetime import datetime
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torch_sparse import spmm
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import pandas as pd
from tqdm import tqdm
from tdc.multi_pred import DTI
from loguru import logger
from torch.utils import data
import sys

sys.path.append("../DeepPurpose-master")
from DeepPurpose import utils, dataset
from DeepPurpose import DTI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *
from Util_Proposed import compute_class_metrics, write_csv, \
    write_csv_withThreshold, check_dir, save_model, DTI_Dataset, dti_df_process, \
    df_data_split, df_data_preprocess, add_virtual_node, \
    generate_sparse_propagation_matrix, convert_features_to_sparse_tensor, \
    load_model
import os
from seed_setting_for_reproducibility import seed_torch


class SparseNGCNLayer(nn.Module):
    """
    Multi-scale Sparse Feature Matrix GCN layer.
    :param in_channels: Number of input features.
    :param out_channels: Number of output features.
    :param iterations: Adjacency matrix power order.
    :param dropout_rate: Dropout rate.
    :param device: Device to run computations on.
    """

    def __init__(self, in_channels, out_channels, iterations, dropout_rate,
                 device):
        super(SparseNGCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.iterations = iterations
        self.dropout_rate = dropout_rate
        self.device = device
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """Define the weight matrices."""
        self.weight_matrix = nn.Parameter(
            torch.Tensor(self.in_channels, self.out_channels)).to(self.device)
        self.bias = nn.Parameter(torch.Tensor(1, self.out_channels)).to(
            self.device)

    def init_parameters(self):
        """Initialize weights using Xavier initialization."""
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, normalized_adjacency_matrix, features):
        """Forward pass."""
        feature_count, _ = torch.max(features["indices"], dim=1)
        feature_count = feature_count + 1
        base_features = spmm(features["indices"], features["values"],
                             feature_count[0],
                             feature_count[1], self.weight_matrix)
        base_features = base_features + self.bias
        base_features = F.dropout(base_features, p=self.dropout_rate,
                                  training=self.training)
        base_features = F.relu(base_features)
        for _ in range(self.iterations - 1):
            base_features = spmm(normalized_adjacency_matrix["indices"],
                                 normalized_adjacency_matrix["values"],
                                 base_features.shape[0],
                                 base_features.shape[0],
                                 base_features)
        return base_features

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.in_channels} -> {self.out_channels})"


class GINLayer(nn.Module):
    """
    Graph Isomorphic Network (GIN) Layer.
    :param in_channels: Number of input features.
    :param out_channels: Number of output features.
    :param epsilon: Small constant for central node feature scaling.
    :param dropout_rate: Dropout rate.
    :param device: Device to run computations on.
    """

    def __init__(self, in_channels, out_channels, epsilon=0.0, dropout_rate=0.5,
                 device=torch.device(
                     'cuda' if torch.cuda.is_available() else 'cpu')):
        super(GINLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.epsilon = epsilon
        self.dropout_rate = dropout_rate
        self.device = device
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels))
        self.dropout = nn.Dropout(dropout_rate)
        self.init_parameters()

    def init_parameters(self):
        """Initialize MLP weights using Xavier initialization."""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, normalized_adjacency_matrix, features):
        """Forward pass."""
        neighbor_features = spmm(normalized_adjacency_matrix["indices"],
                                 normalized_adjacency_matrix["values"],
                                 features.shape[0],
                                 features.shape[0],
                                 features)
        combined_features = (1 + self.epsilon) * features + neighbor_features
        transformed_features = self.mlp(combined_features)
        transformed_features = self.dropout(transformed_features)
        return transformed_features

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.in_channels} -> {self.out_channels})"


def compute_higher_order_adj_onlyGPU(normalized_adjacency_matrix):
    """Compute the second-order adjacency matrix."""
    indices = normalized_adjacency_matrix["indices"]
    values = normalized_adjacency_matrix["values"]
    num_nodes = values.shape[0]
    first_order_adj = torch.sparse_coo_tensor(indices, values,
                                              (num_nodes, num_nodes))
    second_order_adj = torch.sparse.mm(first_order_adj, first_order_adj)
    return {"indices": second_order_adj.indices(),
            "values": second_order_adj.values()}


import torch


def compute_higher_order_adj(normalized_adjacency_matrix):
    """Compute the second-order adjacency matrix."""
    # Extract indices and values from the input dictionary
    if not torch.cuda.is_available():
        return compute_higher_order_adj_onlyGPU(normalized_adjacency_matrix)
    indices = normalized_adjacency_matrix["indices"]
    values = normalized_adjacency_matrix["values"]
    num_nodes = values.shape[0]

    # Move tensors to CPU
    indices_cpu = indices.cpu()
    values_cpu = values.cpu()

    # Create the first-order adjacency matrix on CPU
    first_order_adj = torch.sparse_coo_tensor(indices_cpu, values_cpu,
                                              (num_nodes, num_nodes))

    # Perform the sparse matrix multiplication on CPU
    second_order_adj = torch.sparse.mm(first_order_adj, first_order_adj)

    # Move the result back to the GPU
    second_order_adj_cuda = second_order_adj.to('cuda')

    # Empty the GPU cache to free up unused memory
    torch.cuda.empty_cache()

    # Return the result as a dictionary
    return {
        "indices": second_order_adj_cuda.indices(),
        "values": second_order_adj_cuda.values()
    }


class AttentionCombination(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        # Project all features to a common dimension (e.g., feature_dim // 2)
        self.projection_dim = feature_dim // 2
        self.query_proj = nn.Linear(feature_dim, self.projection_dim)
        self.key_proj = nn.Linear(feature_dim, self.projection_dim)

    def forward(self, features, *agg_features):
        all_features = [features] + list(
            agg_features)  # [features, agg1, agg2, ...]
        num_features = len(all_features)

        # Project Query (original features)
        query = self.query_proj(
            features)  # shape: [batch_size, ..., projection_dim]

        # Project Keys (all features)
        keys = torch.stack([self.key_proj(f) for f in all_features],
                           dim=1)  # [batch_size, num_features, projection_dim]
        # Compute attention scores (dot product + softmax)
        attention_scores = torch.matmul(
            query.unsqueeze(1),  # [batch_size, 1, projection_dim]
            keys.transpose(-1, -2)  # [batch_size, projection_dim, num_features]
        ).squeeze(1)  # [batch_size, num_features]

        attention_weights = torch.softmax(attention_scores,
                                          dim=-1)  # [batch_size, num_features]
        # Stack all features along a new dimension
        stacked_features = torch.stack(all_features,
                                       dim=1)  # [batch_size, num_features, feature_dim]

        # Apply attention weights
        combined_features = torch.sum(
            attention_weights.unsqueeze(-1) * stacked_features,
            dim=1
        )  # [batch_size, feature_dim]

        return combined_features


class HigherOrderGINLayer(nn.Module):
    """
    Higher-Order Graph Isomorphic Network (GIN) Layer.
    :param in_channels: Number of input features.
    :param out_channels: Number of output features.
    :param epsilon: Small constant for central node feature scaling.
    :param dropout_rate: Dropout rate.
    :param device: Device to run computations on.
    """

    def __init__(self, in_channels, out_channels, epsilon=0.0, dropout_rate=0.5,
                 device=torch.device(
                     'cuda' if torch.cuda.is_available() else 'cpu')):
        super(HigherOrderGINLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.epsilon = epsilon
        self.dropout_rate = dropout_rate
        self.device = device
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels))
        self.edge_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels))
        self.first_order_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels))
        self.second_order_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels))
        self.third_order_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels))
        self.fourth_order_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels))
        self.feature_transform = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        # self.weights = nn.Parameter(torch.ones(5))
        self.attentionBasedFeatures = AttentionCombination(
            feature_dim=out_channels)
        self.init_parameters()

    def init_parameters(self):
        """Initialize MLP weights using Xavier initialization."""
        for layer in list(self.mlp) + list(self.edge_mlp) + list(
                self.second_order_mlp) + list(self.third_order_mlp) + list(
                self.fourth_order_mlp):
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, normalized_adjacency_matrix, features, edge_features=None,
                second_order_adj=None, third_order_adj=None,
                fourth_order_adj=None):
        """Forward pass."""
        features = self.feature_transform(features)
        first_order_agg = spmm(normalized_adjacency_matrix["indices"],
                               normalized_adjacency_matrix["values"],
                               features.shape[0],
                               features.shape[0],
                               features)
        first_order_agg = self.first_order_mlp(first_order_agg)
        if edge_features is not None:
            edge_features = self.edge_mlp(edge_features)
            first_order_agg += spmm(normalized_adjacency_matrix["indices"],
                                    normalized_adjacency_matrix["values"],
                                    features.shape[0],
                                    features.shape[0],
                                    edge_features)
        if second_order_adj is not None:
            second_order_agg = spmm(second_order_adj["indices"],
                                    second_order_adj["values"],
                                    features.shape[0],
                                    features.shape[0],
                                    features)
            second_order_agg = self.second_order_mlp(
                second_order_agg) + second_order_agg
        else:
            second_order_agg = torch.zeros_like(first_order_agg)

        if third_order_adj is not None:
            third_order_agg = spmm(third_order_adj["indices"],
                                   third_order_adj["values"],
                                   features.shape[0],
                                   features.shape[0],
                                   features)

            third_order_agg = self.third_order_mlp(
                third_order_agg) + third_order_agg
        else:
            third_order_agg = torch.zeros_like(first_order_agg)

        if fourth_order_adj is not None:
            fourth_order_agg = spmm(fourth_order_adj["indices"],
                                    fourth_order_adj["values"],
                                    features.shape[0],
                                    features.shape[0],
                                    features)

            fourth_order_agg = self.fourth_order_mlp(
                fourth_order_agg) + fourth_order_agg
        else:
            fourth_order_agg = torch.zeros_like(first_order_agg)

        # combined_features = (1 + self.epsilon) * features + first_order_agg + second_order_agg + third_order_agg + fourth_order_agg
        # combined_features = torch.cat([features , first_order_agg , second_order_agg , third_order_agg , fourth_order_agg], dim=-1)
        combined_features = self.attentionBasedFeatures(features,
                                                        first_order_agg,
                                                        second_order_agg,
                                                        third_order_agg,
                                                        fourth_order_agg)
        """
        normalized_weights = torch.softmax(self.weights, dim=0)
        combined_features = (normalized_weights[0] * features +
                    normalized_weights[1] * first_order_agg +
                    normalized_weights[2] * second_order_agg +
                    normalized_weights[3] * third_order_agg +
                    normalized_weights[4] * fourth_order_agg)
        """
        transformed_features = self.mlp(combined_features)
        transformed_features = self.dropout(transformed_features)
        return transformed_features


class HigherOrderMixHop(nn.Module):
    """
    MixHop: Higher-Order Graph Convolutional Architectures.
    :param feature_number: Number of input features.
    :param class_number: Number of target classes.
    :param layers_1: List of layer sizes for the first set of layers.
    :param layers_2: List of layer sizes for the second set of layers.
    :param hidden1: Size of the first hidden layer.
    :param hidden2: Size of the second hidden layer.
    :param dropout: Dropout rate.
    :param device: Device to run computations on.
    """

    def __init__(self, feature_number, class_number=1,
                 layers_1=[32, 32, 32, 32], layers_2=[32, 32, 32, 32],
                 hidden1=64, hidden2=32, dropout=0.1, device=torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')):
        super(HigherOrderMixHop, self).__init__()
        self.layers_1 = layers_1
        self.layers_2 = layers_2
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.feature_number = feature_number
        self.class_number = class_number
        self.dropout = dropout
        self.device = device
        self.calculate_layer_sizes()
        self.setup_layer_structure()
        print(f"{self.__class__}print lne {sys._getframe().f_lineno}")
        self.moleculeAndProteinEmbedding = self.molecule_protein_encoding(
            drug_encoding='MPNN_GCN', target_encoding='CNN_RNN',
            device=device).to(device)  # MPNN_GCN
        self.fusion = nn.Linear(2 * hidden1, hidden1)
        self.second_order_adj = None
        self.third_order_adj = None
        self.fourth_order_adj = None
        # self.deepPurposeModel=None
        # self.config=None

    def calculate_layer_sizes(self):
        """Calculate the sizes of the layers."""
        self.abstract_feature_number_1 = sum(self.layers_1)
        self.abstract_feature_number_2 = sum(self.layers_2)
        self.order_1 = len(self.layers_1)
        self.order_2 = len(self.layers_2)

    def setup_layer_structure(self):
        """Set up the layer structure."""
        self.upper_layers = nn.ModuleList([
            SparseNGCNLayer(self.feature_number, self.layers_1[i - 1], i,
                            self.dropout, self.device) for i in
            range(1, self.order_1 + 1)])
        self.bottom_layers = nn.ModuleList([
            HigherOrderGINLayer(self.abstract_feature_number_1,
                                self.layers_2[i - 1], 32, self.dropout,
                                self.device) for i in
            range(1, self.order_2 + 1)])
        self.bilinear = nn.Bilinear(self.abstract_feature_number_2 * 2,
                                    self.abstract_feature_number_2 * 2,
                                    self.hidden1)
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden1, self.hidden2),
            nn.ELU(),
            nn.Linear(self.hidden2, 1))

    def latent_representations(self, normalized_adjacency_matrix, features):
        """Generate embeddings."""
        abstract_features_1 = torch.cat(
            [self.upper_layers[i](normalized_adjacency_matrix, features) for i
             in range(self.order_1)], dim=1)
        abstract_features_1 = F.dropout(abstract_features_1, self.dropout,
                                        training=self.training)
        if self.second_order_adj is None:
            self.second_order_adj = compute_higher_order_adj(
                normalized_adjacency_matrix)

        if self.third_order_adj is None:
            self.third_order_adj = compute_higher_order_adj(
                self.second_order_adj)

        if self.fourth_order_adj is None:
            self.fourth_order_adj = compute_higher_order_adj(
                self.third_order_adj)
            # self.fourth_order_adj=None;

        abstract_features_2 = torch.cat([self.bottom_layers[i](
            normalized_adjacency_matrix, abstract_features_1,
            second_order_adj=self.second_order_adj,
            third_order_adj=self.third_order_adj,
            fourth_order_adj=self.fourth_order_adj) for i in
                                         range(self.order_2)], dim=1)
        return F.dropout(abstract_features_2, self.dropout,
                         training=self.training)

    def molecule_protein_encoding(self, drug_encoding='MPNN',
                                  target_encoding='CNN', device=torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')):

        if target_encoding == 'CNN':
            config = utils.generate_config(drug_encoding=drug_encoding,
                                           target_encoding=target_encoding,
                                           cls_hidden_dims=[1024, 1024, 512],
                                           train_epoch=5,
                                           LR=0.001,
                                           batch_size=128,
                                           hidden_dim_drug=128,
                                           mpnn_hidden_size=128,
                                           mpnn_depth=3,
                                           cnn_target_filters=[32, 64, 96],
                                           cnn_target_kernels=[4, 8, 12])
        elif target_encoding in ['CNN_RNN', 'Transformer']:
            config = utils.generate_config(
                drug_encoding=drug_encoding,
                target_encoding=target_encoding,
                cls_hidden_dims=[1024, 1024, 512],
                train_epoch=5,
                LR=0.001,
                batch_size=128,
                hidden_dim_drug=128,
                mpnn_hidden_size=128,
                mpnn_depth=3,
                hidden_dim_protein=64,
                mlp_hidden_dims_target=[1024, 256, 64]

            )

        deepPurposeModel = models.model_initialize(**config)
        return deepPurposeModel.embeddings
        # return model.model

    def forward(self, normalized_adjacency_matrix, features, v_d, v_p, idx):

        molecularGraphEmbedding, proteinEmbedding = self.moleculeAndProteinEmbedding(
            v_d, v_p)
        latent_features = self.latent_representations(
            normalized_adjacency_matrix,
            features)  # +mole_protein_basicEmbedding
        feat_p1 = torch.cat([latent_features[idx[0]], molecularGraphEmbedding],
                            dim=-1)
        feat_p2 = torch.cat([latent_features[idx[1]], proteinEmbedding], dim=-1)
        feat = F.elu(self.bilinear(feat_p1, feat_p2))
        feat = F.dropout(feat, self.dropout, training=self.training)
        predictions = self.decoder(feat)
        return predictions, latent_features


class DataDTI(data.Dataset):
    """Dataset for Drug-Target Interaction (DTI) data."""

    def __init__(self, idx_map, labels, df):
        self.labels = labels
        self.idx_map = idx_map
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        idx1 = self.idx_map[str(self.df.iloc[index].Drug_ID)]
        idx2 = self.idx_map[self.df.iloc[index].Target_ID]
        y = self.labels[index]
        return y, (idx1, idx2)


def evaluate_model_performance(model, data_loader, batch_size,
                               propagation_matrix, features):
    """
    Evaluates the model performance using standard classification metrics.

    :param model: Trained model instance.
    :param data_loader: Data loader containing test samples.
    :param batch_size: Number of samples per batch.
    :param propagation_matrix: Precomputed propagation matrix for graph convolution.
    :param features: Node feature matrix.
    :return: Dictionary containing AUROC, AUPRC, and classification metrics.
    """
    model.eval()
    total_batches = len(data_loader)

    y_true = np.empty((total_batches, batch_size))
    y_pred = np.empty((total_batches, batch_size))

    for i in tqdm(range(total_batches), desc='Evaluating Model'):
        # labels, pairs = next(iter(data_loader))  # Fetch batch data
        v_d, v_p, y, idx_1, idx_2, label = next(iter(data_loader))
        with torch.no_grad():
            output, _ = model(propagation_matrix, features, v_d, v_p,
                              [idx_1, idx_2])  # Forward pass

            y_true[i] = label.numpy()
            y_pred[i] = output.flatten().detach().cpu().numpy()

    # Flatten predictions and labels for evaluation
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Binary classification thresholding
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Compute evaluation metrics
    auprc = average_precision_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_pred)
    classification_results = compute_class_metrics(y_true, y_pred_binary)

    # Store results
    classification_results.update({
        'auprc': auprc,
        'auroc': auroc
    })

    return classification_results


def preprocess_dti_dataset(df: pd.DataFrame, threshold=30.0,
                           oversampling: bool = True) -> pd.DataFrame:
    """
    Preprocess Drug-Target Interaction (DTI) dataset by handling missing values,
    creating binary labels based on threshold, and optionally applying oversampling.

    :param df: Input DataFrame containing DTI data.
    :param oversampling: Boolean flag to enable/disable oversampling of negative samples.
    :return: Processed DataFrame with balanced labels if oversampling is enabled.
    """
    df = df.dropna().copy()  # Drop missing values and create a copy to avoid warnings
    df['Drug_ID'] = df['Drug_ID'].astype(str)

    df['Label'] = (df['Y'] > threshold).astype(int)

    neg_samples = df[df['Label'] == 0]
    pos_samples = df[df['Label'] == 1]
    neg_label_num, pos_label_num = len(neg_samples), len(pos_samples)

    # Log sample distribution
    pos_percentage = pos_label_num * 100 / (neg_label_num + pos_label_num)

    if oversampling and neg_label_num < pos_label_num:
        oversampled_neg = pd.concat(
            [neg_samples] * (pos_label_num // neg_label_num), ignore_index=True)
        df = pd.concat([df, oversampled_neg], ignore_index=True)

        # Log updated sample distribution
        neg_label_num, pos_label_num = df[
            'Label'].value_counts().sort_index().tolist()
        pos_percentage = pos_label_num * 100 / (neg_label_num + pos_label_num)

    return df


def train_model(dataset_name, device=torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'), binarizeThreshold=30,
                subsampling=1.0, use_pretrained=False, batch_size=128,
                num_epochs=200, pretrained_dataset_name=None):
    """Train the HigherOrderMixHop model for Drug-Target Interaction (DTI) prediction.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'DAVIS').
        device (torch.device): Device to run the training on (e.g., 'cpu' or 'cuda').
    """
    # Hyperparameters

    learning_rate = 5e-4
    early_stopping_patience = 10

    # Paths for saving results, logs, and models
    if pretrained_dataset_name is None:
        pretrained_dataset_name = dataset_name
    results_base_path = "../results/"
    current_time = datetime.now().strftime("%m-%d-%H-%M")
    experiment_path = f"{results_base_path}output_proposed/{dataset_name}/{current_time}/"
    csv_output_path = f"{results_base_path}output_proposed/{dataset_name}/"
    model_save_path = f"{results_base_path}output/model/"
    log_file_path = f"{experiment_path}/train.log"
    model_save_name = f"proposedMixhopHIORDER_{dataset_name}_epoch{num_epochs}.pt"
    pretrained_model_name = f"proposedMixhopHIORDER_{pretrained_dataset_name}_epoch{num_epochs}.pt"
    # Create directories if they don't exist
    check_dir(experiment_path)
    check_dir(csv_output_path)
    check_dir(model_save_path)
    log_fd = logger.add(log_file_path)

    # Load and preprocess the dataset
    dti_data = DTI(name=dataset_name)
    print(
        f'data binding affinity before Binarization minimum value: {np.min(dti_data.get_data().Y)}, maximum value :{np.max(dti_data.get_data().Y)}, summary: {dti_data.get_data().Y.describe()}')

    if dataset_name in "DAVIS":
        dti_data.convert_to_log(form='binding')
        print(
            f'data binding affinity after logarithm minimum value: {np.min(dti_data.get_data().Y)}, maximum value :{np.max(dti_data.get_data().Y)} summary: {dti_data.get_data().Y.describe()}')
        dti_data.binarize(threshold=binarizeThreshold,
                          order='descending')  # 7, change to 30
    elif dataset_name == "BindingDB_Kd":
        dti_data.convert_to_log(form='binding')
        print(
            f'data binding affinity after logarithm minimum value: {np.min(dti_data.get_data().Y)}, maximum value :{np.max(dti_data.get_data().Y)} summary: {dti_data.get_data().Y.describe()}')
        dti_data.binarize(threshold=binarizeThreshold,
                          order='descending')  # 7.6 #9.5 gives the best results

    elif dataset_name == "KIBA":
        dti_data.binarize(threshold=binarizeThreshold, order='descending')
        print(
            f'data binding affinity after logarithm minimum value: {np.min(dti_data.get_data().Y)}, maximum value :{np.max(dti_data.get_data().Y)} summary: {dti_data.get_data().Y.describe()}')
    else:
        logger.error(f"dataset {dataset_name} is not supported")
        return

    all_data = dti_data.get_data()
    print(
        f'data binding affinity after Binarization minimum value: {np.min(all_data.Y)}, maximum value :{np.max(all_data.Y)} summary: {dti_data.get_data().Y.describe()}')
    # exit(0)

    processed_data = df_data_preprocess(all_data)
    logger.info(f"Dataset: {dataset_name}\n{processed_data}")

    # Create mappings for drugs and targets
    """
    
    unique_entities = processed_data['Drug_ID'].tolist() + processed_data['Target_ID'].tolist()
    unique_entities = list(set(unique_entities))
    entity_to_index = {entity: idx for idx, entity in enumerate(unique_entities)}
    #idx = np.concatenate((processed_data['Drug_ID'].unique(), processed_data['Target_ID'].unique()))
    #idx_map = {j: i for i, j in enumerate(idx)}

    # Create sparse feature matrix (one-hot encoding)
    num_entities = len(unique_entities)
    feature_matrix = np.eye(num_entities)
    sparse_feature_matrix = convert_features_to_sparse_tensor(feature_matrix, device)

    # Create adjacency matrix for the graph
    edges = processed_data[['Drug_ID', 'Target_ID']].values
    edge_indices = np.array(list(map(entity_to_index.get, edges.flatten()))).reshape(edges.shape)
    adjacency_matrix = sp.coo_matrix((np.ones(edge_indices.shape[0]), (edge_indices[:, 0], edge_indices[:, 1])),
                                     shape=(num_entities, num_entities), dtype=np.float32)
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T.multiply(adjacency_matrix.T > adjacency_matrix) - adjacency_matrix.multiply(adjacency_matrix.T > adjacency_matrix)
    propagation_matrix = generate_sparse_propagation_matrix(adjacency_matrix, device)
    """
    idx = np.concatenate((processed_data['Drug_ID'].unique(),
                          processed_data['Target_ID'].unique()))
    entity_idx_map = {j: i for i, j in enumerate(idx)}

    edges_unordered = processed_data[['Drug_ID', 'Target_ID']].values
    idx_total = len(idx)
    features = np.eye(idx_total + 1)  # Drug_ID + Target_ID + 1 virtual node
    # features = np.vstack([features, np.zeros((1, idx_total), dtype=np.float32)]) #because of virtual node
    # features = np.vstack([features, np.zeros((1, idx_total), dtype=np.float32)])  # (N+1, N)

    assert features.shape[0] == idx_total + 1  # Sanity check
    sparse_feature_matrix = convert_features_to_sparse_tensor(features, device)
    edges = np.array(
        list(map(entity_idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(idx), len(idx)),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    propagation_matrix = generate_sparse_propagation_matrix(adj, device)

    # Split dataset into train, validation, and test sets
    # dataset_split = dti_data.get_split(method='random', seed=42, frac=[0.7, 0.1, 0.2])
    # train_data = preprocess_dti_dataset(dataset_split['train'],threshold=binarizeThreshold)
    # val_data = preprocess_dti_dataset(dataset_split['valid'], threshold=binarizeThreshold)
    # test_data = preprocess_dti_dataset(dataset_split['test'], threshold=binarizeThreshold)
    df_train, df_valid, df_test = df_data_split(processed_data,
                                                frac=[0.7 * subsampling, 0.1,
                                                      0.2 / subsampling])
    df_train = dti_df_process(df_train, target_encoding='CNN_RNN')
    df_valid = dti_df_process(df_valid, target_encoding='CNN_RNN')
    df_test = dti_df_process(df_test, target_encoding='CNN_RNN')

    # Create data loaders
    """
    
    train_loader = data.DataLoader(DataDTI(entity_to_index, train_data.Label.values, train_data),
                                   batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = data.DataLoader(DataDTI(entity_to_index, val_data.Label.values, val_data),
                                 batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = data.DataLoader(DataDTI(entity_to_index, test_data.Label.values, test_data),
                                  batch_size=batch_size, shuffle=False, drop_last=True)
    """
    train_params = {'batch_size': batch_size,
                    'shuffle': True,
                    # 'num_workers': 6,
                    'drop_last': True,
                    # 'pin_memory':True,
                    # 'prefetch_factor':2,
                    'collate_fn': utils.mpnn_collate_func
                    }

    test_params = {'batch_size': batch_size,
                   'shuffle': False,
                   # 'num_workers': 6,
                   'drop_last': True,
                   'collate_fn': utils.mpnn_collate_func
                   }
    train_dataset = DTI_Dataset(entity_idx_map, df_train)
    train_loader = data.DataLoader(train_dataset, **train_params)

    valid_dataset = DTI_Dataset(entity_idx_map, df_valid)
    val_loader = data.DataLoader(valid_dataset, **test_params)

    test_dataset = DTI_Dataset(entity_idx_map, df_test)
    test_loader = data.DataLoader(test_dataset, **test_params)
    # Initialize model and optimizer
    num_features = sparse_feature_matrix["dimensions"][1]
    model = HigherOrderMixHop(num_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if not use_pretrained:
        # Training loop
        max_validation_auc = 0
        no_improvement_counter = 0
        start_time = time.time()
        logger.info('Starting training...')

        for epoch in range(num_epochs):
            model.train()
            epoch_start_time = time.time()
            epoch_loss = 0
            train_predictions = []
            train_labels = []
            batch_total = len(train_loader)

            # for batch_idx, (labels, pairs) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            for i in tqdm(range(batch_total), f"train epoch{epoch + 1}"):
                v_d, v_p, y, idx_1, idx_2, label = next(iter(train_loader))

                labels = label.to(device)
                optimizer.zero_grad()
                predictions, _ = model(propagation_matrix,
                                       sparse_feature_matrix, v_d, v_p,
                                       [idx_1, idx_2])
                loss = F.binary_cross_entropy_with_logits(predictions.squeeze(),
                                                          labels.float())
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                train_predictions.extend(
                    predictions.detach().cpu().numpy().flatten())
                train_labels.extend(labels.cpu().numpy().flatten())
                if len(np.unique(train_labels)) < 2:
                    print(
                        "Warning: Only one class present in y_true. Skipping ROC AUC calculation.")
                    exit(0)

                # Log batch loss
                write_csv(f"{csv_output_path}proposedMixhopHIORDER_loss.csv",
                          {'epoch': epoch, 'batch': 0, 'loss': loss.item(),
                           'avg_loss': epoch_loss / (0 + 1)})

            # Calculate training metrics

            train_auc = roc_auc_score(train_labels, train_predictions)
            logger.info(
                f"Epoch {epoch + 1}: Training AUC = {train_auc:.4f}, Loss = {epoch_loss / len(train_loader):.4f}")

            # Evaluate on validation set
            validation_metrics = evaluate_model_performance(model, val_loader,
                                                            batch_size,
                                                            propagation_matrix,
                                                            sparse_feature_matrix)
            validation_metrics['epoch'] = epoch
            validation_metrics['epoch_loss'] = epoch_loss / len(train_loader)
            write_csv(f"{csv_output_path}proposedMixhopHIORDER_val_metrics.csv",
                      validation_metrics)
            logger.info(f"Validation Metrics: {validation_metrics}")

            # Early stopping
            if validation_metrics['auroc'] > max_validation_auc:
                max_validation_auc = validation_metrics['auroc']
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1
                if no_improvement_counter == early_stopping_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch + 1} due to no improvement in validation AUC.")
                    break

            logger.info(
                f"Epoch {epoch + 1} completed in {time.time() - epoch_start_time:.2f}s")

        # Save the trained model

        save_model(model, f"{model_save_path}{model_save_name}")
        logger.info(f"Model saved to {model_save_path}{model_save_name}")
        logger.info(f"Total training time: {time.time() - start_time:.2f}s")
    else:

        model = load_model(model, f"{model_save_path}{pretrained_model_name}",
                           device=device)

    # Evaluate on test set
    test_metrics = evaluate_model_performance(model, test_loader, batch_size,
                                              propagation_matrix,
                                              sparse_feature_matrix)
    write_csv_withThreshold(
        f"{csv_output_path}proposed_mixhopHIORDER_{dataset_name}_test_metrics.csv",
        test_metrics, threshold_val=binarizeThreshold)
    logger.info(f"Test Metrics: {test_metrics}")

    # Clean up logging
    logger.remove(log_fd)
