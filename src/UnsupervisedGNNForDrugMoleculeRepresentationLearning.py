import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Planetoid
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import numpy as np
from itertools import combinations
from sklearn.metrics import mutual_info_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mutual_info_score
from itertools import combinations
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import MLP, GINConv, global_add_pool

class SynergyCalculator(nn.Module):
    def __init__(self, n_bins=10, margin=1.0, init_alpha=0.4, init_beta=0.3):
        super(SynergyCalculator, self).__init__()
        self.n_bins = n_bins
        self.margin = margin  # For contrastive loss

        # Initialize learnable logits for alpha and beta (bounded in [0,1] via sigmoid)
        self.logit_alpha = nn.Parameter(torch.tensor(init_alpha).logit())
        self.logit_beta = nn.Parameter(torch.tensor(init_beta).logit())

    def get_weights(self):
        # Convert logits to probabilities in [0,1], then normalize
        alpha = torch.sigmoid(self.logit_alpha)
        beta = torch.sigmoid(self.logit_beta)
        total = alpha + beta
        alpha = alpha / total
        beta = beta / total
        gamma = 1.0 - alpha - beta  # Synergy weight
        return alpha, beta, gamma

    def discretize(self, signal):
        """Discretize signal into bins"""
        with torch.no_grad():
            bins = torch.linspace(signal.min(), signal.max(), self.n_bins + 1, device=signal.device)
            return torch.bucketize(signal, bins[1:-1])

    def calculate_pmi_loss(self, embeddings, edge_index):
        """PMI loss using dense adjacency approximation"""
        num_nodes = embeddings.size(0)

        adj = torch.zeros((num_nodes, num_nodes), device=embeddings.device)
        adj[edge_index[0], edge_index[1]] = 1
        adj[range(num_nodes), range(num_nodes)] = 1  # self-loops

        eps = 1e-10
        total_edges = adj.sum().float()
        p_observed = adj / (total_edges + eps)

        node_degrees = adj.sum(dim=1).float()
        p_expected = torch.outer(node_degrees, node_degrees) / (total_edges ** 2 + eps)

        pmi = torch.log(p_observed + eps) - torch.log(p_expected + eps)

        norm_emb = F.normalize(embeddings, p=2, dim=1)
        cosine_sim = torch.mm(norm_emb, norm_emb.t())

        return -torch.mean(pmi * cosine_sim)

    def calculate_contrastive_loss(self, embeddings, edge_index):
        """Contrastive loss via cosine similarity"""
        row, col = edge_index
        pos_sim = F.cosine_similarity(embeddings[row], embeddings[col])

        neg_nodes = torch.randint(0, embeddings.size(0), (row.size(0),))
        neg_sim = F.cosine_similarity(embeddings[row], embeddings[neg_nodes])

        return F.relu(self.margin - pos_sim + neg_sim).mean()

    def calculate_synergy(self, embeddings, node_set, training=False):
        """Synergy calculation with discrete info theory proxy"""
        if len(node_set) < 3:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=training)

        if training:
            embeddings.retain_grad()

        with torch.no_grad():
            data = embeddings[node_set].cpu().numpy().T
            sources = data[:, :-1]
            target = data[:, -1]

            sources_discrete = np.array([
                self.discretize(torch.tensor(col)).numpy()
                for col in sources.T
            ]).T
            target_discrete = self.discretize(torch.tensor(target)).numpy()

            combo_sources = np.array([
                int("".join(str(d) for d in row), self.n_bins)
                for row in sources_discrete
            ])
            total_info = mutual_info_score(combo_sources, target_discrete)

            unique_info = max(
                mutual_info_score(sources_discrete[:, i], target_discrete)
                for i in range(sources_discrete.shape[1])
            )

            redundant_info = max(
                mutual_info_score(sources_discrete[:, i], sources_discrete[:, j])
                for i, j in combinations(range(sources_discrete.shape[1]), 2)
            ) if sources_discrete.shape[1] >= 2 else 0.0

            synergy = max(0.0, total_info - unique_info - redundant_info)

        proxy = torch.tensor(synergy, device=embeddings.device, requires_grad=training)

        if training:
            def backward_hook(grad):
                if embeddings.grad is None:
                    embeddings.grad = torch.zeros_like(embeddings)
                embeddings.grad[node_set] += grad * 0.01

            proxy.register_hook(backward_hook)

        return proxy

    def combined_loss(self, embeddings, node_sets, edge_index, training=False):
        """Total loss using learnable weights"""
        synergy_loss = -torch.stack([
            self.calculate_synergy(embeddings, nodes, training)
            for nodes in node_sets
        ]).mean()

        contrastive_loss = self.calculate_contrastive_loss(embeddings, edge_index)
        pmi_loss = self.calculate_pmi_loss(embeddings, edge_index)

        alpha, beta, gamma = self.get_weights()
        total_loss = gamma * synergy_loss + alpha * contrastive_loss + beta * pmi_loss

        return {
            'total_loss': total_loss,
            'synergy_loss': synergy_loss,
            'contrastive_loss': contrastive_loss,
            'pmi_loss': pmi_loss,
            'alpha': alpha.item(),
            'beta': beta.item(),
            'gamma': gamma.item()
        }

def contrastive_loss(embeddings, edge_index, margin=1.0):
    """Push connected nodes closer, disconnected nodes apart"""
    row, col = edge_index
    pos_sim = F.cosine_similarity(embeddings[row], embeddings[col])

    # Negative sampling
    neg_nodes = torch.randint(0, embeddings.size(0), (row.size(0),))
    neg_sim = F.cosine_similarity(embeddings[row], embeddings[neg_nodes])

    return torch.mean(F.relu(margin - pos_sim + neg_sim))
class SynergyGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, out_dim)  # Added third layer
        self.dropout = nn.Dropout(0.6)  # Increased dropout

    def forward(self, x, edge_index):
        x = F.leaky_relu(self.conv1(x, edge_index))  # Changed activation
        x = self.dropout(x)
        x = F.leaky_relu(self.conv2(x, edge_index))
        return self.conv3(x, edge_index)

def evaluate_embeddings(embeddings, labels, n_neighbors=5):
    """Unsupervised evaluation metrics"""
    emb_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # 1. Silhouette Score
    try:
        silhouette = silhouette_score(emb_np, labels_np)
    except:
        silhouette = -1

    # 2. Calinski-Harabasz Index
    ch_score = calinski_harabasz_score(emb_np, labels_np)

    # 3. KNN Consistency
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(emb_np)
    distances, indices = nbrs.kneighbors(emb_np)
    knn_consistency = np.mean([np.mean(labels_np[indices[i]] == labels_np[i])
                             for i in range(len(indices))])

    return {
        'silhouette': silhouette,
        'calinski_harabasz': ch_score,
        'knn_consistency': knn_consistency
    }

from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, MessagePassing

# Define a simple MPNN layer
class MPNNLayer(MessagePassing):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MPNNLayer, self).__init__(aggr='add')  # Using 'add' aggregation
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Linear layers for transforming features
        self.lin_msg = nn.Linear(in_dim, hidden_dim)  # Linear layer to transform input features
        self.lin_agg = nn.Linear(hidden_dim, out_dim)  # Linear layer to transform aggregated features

    def forward(self, x, edge_index):
        # Propagate the input features through message passing
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # Apply the linear layer to the neighboring node features
        return self.lin_msg(x_j)

    def aggregate(self, inputs, index, dim_size=None):
        # Aggregate the messages using sum
        return torch.sum(inputs, dim=0)

    def update(self, aggr_out):
        # Apply the linear layer to the aggregated messages
        return self.lin_agg(aggr_out)


# PAGNN Layer (simplified version using position-aware encodings)
class PAGNNConv(MessagePassing):
    def __init__(self, in_dim, out_dim):
        super(PAGNNConv, self).__init__(aggr='add')
        self.lin = nn.Linear(in_dim, out_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, in_dim))  # Learnable position encoding

    def forward(self, x, edge_index):
        # Add positional encoding to node features
        x = x + self.positional_encoding
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j  # Message passing with positional encoding added

    def update(self, aggr_out, x):
        return self.lin(aggr_out)  # Node update

import torch
import torch.nn as nn

class LearnableLossWeights(nn.Module):
    def __init__(self, init_alpha=0.5, init_beta=0.4):
        super().__init__()
        # Use softplus or sigmoid to ensure positivity or bounded range
        self.logit_alpha = nn.Parameter(torch.tensor(init_alpha).logit())  # if using sigmoid
        self.logit_beta = nn.Parameter(torch.tensor(init_beta).logit())

    def get_weights(self):
        alpha = torch.sigmoid(self.logit_alpha)
        beta = torch.sigmoid(self.logit_beta)
        # Optionally normalize to ensure α + β ≤ 1
        total = alpha + beta
        alpha = alpha / total
        beta = beta / total
        return alpha, beta

class DeeperSynergyGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, conv_type='GCN', output_type='single',concat_all=True):
        super().__init__()
        self.conv_type = conv_type
        self.output_type = output_type
        print(f'conve type: {conv_type}')
        self.concat_all=concat_all
        self.linear=nn.Linear(in_dim, hidden_dim);

        if conv_type == 'GAT':
            self.conv1 = GATConv(in_dim, hidden_dim, heads=4, concat=True)
            self.linear=nn.Linear(in_dim, hidden_dim * 4);
            self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True)
            self.conv3 = GATConv(hidden_dim * 4, out_dim, heads=1, concat=False)

        elif conv_type == 'GCN':
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, out_dim)

        elif conv_type == 'GIN':
            self.conv1 = GINConv(nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
            self.conv2 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
            self.conv3 = GINConv(nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))

        elif conv_type == 'SAGE':
            self.conv1 = SAGEConv(in_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.conv3 = SAGEConv(hidden_dim, out_dim)

        elif conv_type == 'MPNN':
            self.conv1 = MPNNLayer(in_dim, hidden_dim, out_dim)
            self.conv2 = MPNNLayer(hidden_dim, hidden_dim, out_dim)
            self.conv3 = MPNNLayer(hidden_dim, hidden_dim, out_dim)

        elif conv_type == 'PAGNN':
            self.conv1 = PAGNNConv(in_dim, hidden_dim)
            self.conv2 = PAGNNConv(hidden_dim, hidden_dim)
            self.conv3 = PAGNNConv(hidden_dim, out_dim)
        elif conv_type == 'ALL':
            self.gats = nn.ModuleList([
                GATConv(in_dim, hidden_dim, heads=4, concat=True),
                GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True),
                GATConv(hidden_dim * 4, out_dim, heads=1, concat=False)
            ])
            self.linear=nn.Linear(in_dim, hidden_dim * 4);

            self.gcns = nn.ModuleList([
                GCNConv(in_dim, hidden_dim),
                GCNConv(hidden_dim, hidden_dim),
                GCNConv(hidden_dim, out_dim)
            ])

            self.gins = nn.ModuleList([
                GINConv(nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))),
                GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))),
                GINConv(nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))
            ])

            if self.concat_all:
                self.fusion = nn.Linear(out_dim*3 , out_dim)  # if concatenating 3 paths
            else:
                self.fusion = nn.Linear(out_dim , out_dim)



        else:
            raise ValueError(f"Unsupported convolution type: {conv_type}")

        self.dropout = nn.Dropout(0.6)  # More dropout


    def forward(self, x, edge_index):
        # List to hold all convolution outputs
        conv_outputs = []

        if self.conv_type in ['GAT', 'GIN', 'PAGNN']:
            x1 = F.leaky_relu(self.conv1(x, edge_index)+self.linear(x), negative_slope=0.2)
            x1 = self.dropout(x1)
            x2 = F.leaky_relu(self.conv2(x1, edge_index), negative_slope=0.2)
            conv_outputs.append(self.conv3(x2, edge_index))
        if self.conv_type in ['MPNN']:
            x1 = F.leaky_relu(self.conv1(x, edge_index)+self.linear(x), negative_slope=0.2)
            x1 = self.dropout(x1)

            # Apply the second MPNN layer
            x2 = F.leaky_relu(self.conv2(x1, edge_index), negative_slope=0.2)
            x2 = self.dropout(x2)
            conv_outputs.append(self.conv3(x2, edge_index))
        if self.conv_type in ['ALL']:
            x_gat = F.elu(self.gats[0](x, edge_index)+self.linear(x))
            x_gat = F.elu(self.gats[1](x_gat, edge_index))
            x_gat = self.gats[2](x_gat, edge_index)

            # GCN path
            x_gcn = F.relu(self.gcns[0](x, edge_index))
            x_gcn = F.relu(self.gcns[1](x_gcn, edge_index))
            x_gcn = self.gcns[2](x_gcn, edge_index)

            # GIN path
            x_gin = F.relu(self.gins[0](x, edge_index))
            x_gin = F.relu(self.gins[1](x_gin, edge_index))
            x_gin = self.gins[2](x_gin, edge_index)


            if self.concat_all:
                x_combined = torch.cat([x_gat, x_gcn, x_gin], dim=1)  # shape: (num_nodes, out_dim*3)
            else:
                x_combined = x_gat+ x_gcn + x_gin
            x_combined = self.dropout(x_combined)
            out = self.fusion(x_combined)  # shape: (num_nodes, out_dim)
            return out
        else:
            x1 = F.relu(self.conv1(x, edge_index))
            x1 = self.dropout(x1)
            x2 = F.relu(self.conv2(x1, edge_index))
            conv_outputs.append(self.conv3(x2, edge_index))

        if self.output_type == 'ALL':
            # Concatenate all outputs from different layers
            return torch.cat(conv_outputs, dim=1)  # Concatenate along the feature dimension
        else:
            # Return just the final layer output
            return conv_outputs[-1]

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return self.mlp(x)

def train(model, data, optimizer, synergy_calc, randomNodeSet=32, epochs=100):
    model.train()
    best_loss = float('inf')
    no_improve_count = 0
    patience = 10  # Stop if no improvement for this many epochs

    for epoch in range(epochs):
        optimizer.zero_grad()
        embeddings = model(data.x, data.edge_index)

        # Sample node triplets
        node_sets = [torch.randperm(data.num_nodes)[:3] for _ in range(randomNodeSet)]

        # Calculate combined loss
        loss_dict = synergy_calc.combined_loss(
            embeddings,
            node_sets,
            data.edge_index,
            training=True
        )

        total_loss = loss_dict['total_loss']
        total_loss.backward()
        optimizer.step()

        current_loss = total_loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch}:")
            print(f"  Total Loss: {current_loss:.4f}")
            print(f"  Synergy Loss: {loss_dict['synergy_loss'].item():.4f}")
            print(f"  Contrastive Loss: {loss_dict['contrastive_loss'].item():.4f}")
            print(f"  PMI Loss: {loss_dict['pmi_loss'].item():.4f}")

        if no_improve_count >= patience:
            print(f"Stopping early at epoch {epoch} — no improvement in {patience} epochs.")
            break



import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def test12(model, data, synergy_calc, n_tests=100):
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)

        # 1. Calculate synergy metrics
        synergies = []
        for _ in range(n_tests):
            nodes = torch.randperm(data.num_nodes)[:3]
            synergy = synergy_calc.calculate_synergy(embeddings, nodes, training=False)
            synergies.append(synergy.item())
        avg_synergy = np.mean(synergies) if synergies else 0.0

        # 2. Unsupervised clustering metrics
        eval_metrics = evaluate_embeddings(embeddings, data.y)

        # 3. Visualize embeddings with t-SNE
        visualize_embeddings(embeddings, data.y, title="Cora Node Embeddings")

        print("\nTest Results:")
        print(f"Average Synergy: {avg_synergy:.4f}")
        print(f"Silhouette Score: {eval_metrics['silhouette']:.4f}")
        print(f"Calinski-Harabasz Index: {eval_metrics['calinski_harabasz']:.4f}")
        print(f"KNN Consistency: {eval_metrics['knn_consistency']:.4f}")

        return {
            'synergy': avg_synergy,
            **eval_metrics
        }

import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

def link_prediction_eval(embeddings, edge_index, num_negative_samples=10000, return_details=False):
    from torch_geometric.utils import negative_sampling

    # 1. Sample negative edges
    pos_edge_index = edge_index
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=embeddings.size(0),
        num_neg_samples=num_negative_samples
    )

    def dot_product(u, v):
        return (u * v).sum(dim=-1)

    # 2. Compute scores for positive and negative edges
    pos_scores = dot_product(embeddings[pos_edge_index[0]], embeddings[pos_edge_index[1]])
    neg_scores = dot_product(embeddings[neg_edge_index[0]], embeddings[neg_edge_index[1]])

    # 3. Combine for labels and predictions
    y_true = torch.cat([
        torch.ones(pos_scores.size(0)),
        torch.zeros(neg_scores.size(0))
    ]).cpu().numpy()

    y_scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
    y_pred = (y_scores >= 0).astype(int)  # Binary prediction based on threshold

    # 4. Classification Metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auroc = roc_auc_score(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)

    # 5. Confusion Matrix for Specificity
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0

    # 6. Return metrics
    metrics = {
        "LP_accuracy": acc,
        "LP_precision": precision,
        "LP_recall": recall,
        "LP_f1": f1,
        "LP_auroc": auroc,
        "LP_aupr": aupr,
        "LP_specificity": specificity,
    }

    return metrics



def graph_reconstruction_loss(embeddings, edge_index, num_nodes):
    # Compute full similarity matrix
    logits = torch.matmul(embeddings, embeddings.t())  # [N, N]
    adj_pred = torch.sigmoid(logits)

    # Build ground truth adjacency matrix
    adj_true = torch.zeros((num_nodes, num_nodes), device=embeddings.device)
    adj_true[edge_index[0], edge_index[1]] = 1
    adj_true[edge_index[1], edge_index[0]] = 1  # Assuming undirected graph

    # Binary cross-entropy loss
    loss = F.binary_cross_entropy(adj_pred, adj_true)
    return loss.item()
import torch

def pairwise_distance_correlation(embeddings, edge_index, metric='cosine'):
    num_nodes = embeddings.size(0)

    # Compute pairwise similarity matrix
    if metric == 'cosine':
        normed = F.normalize(embeddings, p=2, dim=1)
        sim_matrix = torch.mm(normed, normed.t())
    elif metric == 'dot':
        sim_matrix = torch.mm(embeddings, embeddings.t())
    elif metric == 'euclidean':
        sim_matrix = -torch.cdist(embeddings, embeddings, p=2)
    else:
        raise ValueError("Unsupported metric")

    # Build binary adjacency matrix
    adj = torch.zeros((num_nodes, num_nodes), device=embeddings.device)
    adj[edge_index[0], edge_index[1]] = 1
    adj[edge_index[1], edge_index[0]] = 1  # Assuming undirected

    # Flatten and compute correlation
    sim_flat = sim_matrix.flatten().cpu()
    adj_flat = adj.flatten().cpu()

    # Pearson correlation
    corr = torch.corrcoef(torch.stack([adj_flat, sim_flat]))[0, 1]
    return corr.item()

def visualize_embeddings(embeddings, labels, title="Embeddings"):
    # Convert to numpy
    emb_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Reduce to 2D using t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    emb_2d = tsne.fit_transform(emb_np)

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=labels_np,
        cmap='viridis',
        alpha=0.6,
        s=10
    )

    # Add legend and labels
    plt.colorbar(scatter, label='Class')
    plt.title(f"{title} (t-SNE projection)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    # Save and show
    plt.savefig("cora_embeddings.png", dpi=300, bbox_inches='tight')
    plt.show()

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def test(model, data, synergy_calc, n_tests=100):
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)

        # 1. Synergy metric
        synergies = []
        for _ in range(n_tests):
            nodes = torch.randperm(data.num_nodes)[:3]
            synergy = synergy_calc.calculate_synergy(embeddings, nodes, training=False)
            synergies.append(synergy.item())
        avg_synergy = np.mean(synergies) if synergies else 0.0

        # 2. Clustering metrics (optional)
        eval_metrics = evaluate_embeddings(embeddings, data.y)

        # 3. Node classification (Multiclass)
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings.cpu().numpy(), data.y.cpu().numpy(), test_size=0.2, random_state=42
        )
        clf = LogisticRegression(max_iter=500, multi_class='ovr')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Classification metrics (multiclass macro-avg)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        # 4. Link prediction
        link_prediction_evaluations = link_prediction_eval(
            embeddings, data.edge_index, return_details=True
        )



        # 5. Graph reconstruction
        graphRecons_BCEloss = graph_reconstruction_loss(embeddings, data.edge_index, embeddings.shape[0])

        # 6. Pairwise distance correlations
        corr_cosine = pairwise_distance_correlation(embeddings, data.edge_index, metric='cosine')
        corr_dot = pairwise_distance_correlation(embeddings, data.edge_index, metric='dot')
        corr_euclidean = pairwise_distance_correlation(embeddings, data.edge_index, metric='euclidean')

        # Final report
        results = {
            # Synergy
            'synergy': avg_synergy,

            # Node Classification
            'node_cls_accuracy': accuracy,
            'node_cls_precision': precision,
            'node_cls_recall (sensitivity)': recall,
            'node_cls_f1': f1,

            # Link Prediction
            **link_prediction_evaluations,

            # Other graph metrics
            'graph_reconstruction_bce_loss': graphRecons_BCEloss,
            'cosine-adj_corr': corr_cosine,
            'dot-adj_corr': corr_dot,
            'euclidean-adj_corr': corr_euclidean,

            # Optional clustering
            **eval_metrics
        }

        # Print nicely formatted
        print("\n📊 Evaluation Summary:")
        for k, v in results.items():
            print(f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}")

        return results



import csv
import torch
from datetime import datetime

def save_results_to_csv(model, test_results, conv_type, in_dim, hidden_dim, out_dim, dropout_rate, file_name="./result/unsupervisedresult.csv"):
    fieldnames = ['convolution_type', 'in_dim', 'hidden_dim', 'out_dim', 'dropout_rate', 'test_metric', 'value', 'timestamp']

    # Ensure file exists and has headers
    try:
        with open(file_name, mode='r', newline='') as file:
            pass
    except FileNotFoundError:
        with open(file_name, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

    # Save model
    torch.save(model.state_dict(), "synergy_gnn_cora.pth")

    # Timestamp
    now = datetime.now()
    timestamp_str = now.strftime("%b-%d %H:%M")  # e.g., "Apr-05 15:42"

    print(f"\nModel saved with test metrics:")
    with open(file_name, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        for k, v in test_results.items():
            if isinstance(v, dict):  # Handling nested dictionary
                for subk, subv in v.items():
                    # Check if the value is numeric before formatting
                    value = f"{subv:.4f}" if isinstance(subv, (int, float)) else str(subv)
                    writer.writerow({
                        'convolution_type': conv_type,
                        'in_dim': in_dim,
                        'hidden_dim': hidden_dim,
                        'out_dim': out_dim,
                        'dropout_rate': dropout_rate,
                        'test_metric': f"{k}_{subk}",
                        'value': value,
                        'timestamp': timestamp_str
                    })
                    print(f"{k}_{subk}: {value}")
            else:
                # Check if the value is numeric before formatting
                value = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
                writer.writerow({
                    'convolution_type': conv_type,
                    'in_dim': in_dim,
                    'hidden_dim': hidden_dim,
                    'out_dim': out_dim,
                    'dropout_rate': dropout_rate,
                    'test_metric': k,
                    'value': value,
                    'timestamp': timestamp_str
                })
                print(f"{k}: {value}")

        # Add one separator line
        writer.writerow({
            'convolution_type': '---',
            'in_dim': '',
            'hidden_dim': '',
            'out_dim': '',
            'dropout_rate': '',
            'test_metric': '',
            'value': '',
            'timestamp': ''
        })

"""
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
"""
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    datasets={'Cora','Pubmed','Citeseer'}
    datasets={'Cora','Citeseer'}
    data_name='Citeseer'


    for data_name in datasets:

        dataset = Planetoid(root='F:/AbbasFromDesktop/ZKNU2024/students guide 2024/Thesis2024-2025/pytorch_geometric-master/data', name=data_name)
        data = dataset[0].to(device)
        print(f'data set used: {data_name}')

        conv_types ={'GAT','GCN','SAGE','GIN','PAGNN', 'ALL'}  # Example convolution type
        #conv_types ={ 'ALL'}  # Example convolution type
        for conv_type in conv_types:

            in_dim = dataset.num_features  # Example input dimension
            hidden_dim = 128  # Example hidden dimension
            out_dim = dataset.num_classes # Example output dimension
            dropout_rate = 0.6  # Example dropout rate

            model = DeeperSynergyGNN(in_dim, hidden_dim, 64,conv_type=conv_type,concat_all=False).to(device)

            synergy_calc = SynergyCalculator( )  # 30% weight to PMI)
            optimizer = torch.optim.Adam(list(model.parameters()) + list(synergy_calc.parameters()), lr=0.005, weight_decay=5e-4)

            train(model, data, optimizer, synergy_calc,randomNodeSet=32, epochs=500)
            test_results = test(model, data, synergy_calc)
            save_results_to_csv(model, test_results, conv_type, in_dim, hidden_dim, out_dim, dropout_rate,file_name=f"./result/{data_name}_unsupervisedresult.csv")
            torch.save(model.state_dict(), "synergy_gnn_cora.pth")
            print(f"\nModel saved with test metrics:")
            #for k, v in test_results.items():
             #   print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
