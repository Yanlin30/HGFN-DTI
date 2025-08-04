import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool  # Example GNN layers
import torch.nn.functional as F

class ProteinGraphEncoder(nn.Sequential):
    def __init__(self, encoding, **config):
        super(ProteinGraphEncoder, self).__init__()
        assert encoding == 'protein', "Graph encoder only for proteins"

        # Graph construction parameters
        self.graph_type = config.get('graph_type', 'distance')  # 'distance' or 'sequence'
        self.threshold_distance = config.get('threshold_distance', 8.0)  # Ångstroms

        # GNN architecture parameters
        self.gnn_layers = config.get('gnn_layers', 3)
        self.gnn_hidden_size = config.get('gnn_hidden_size', 128)
        self.gnn_heads = config.get('gnn_heads', 4)  # For GAT

        # Node embedding layer (replaces transformer's Embeddings)
        self.node_emb = nn.Embedding(config['input_dim_protein'],
                                    self.gnn_hidden_size)

        # GNN layers (e.g., Graph Attention Networks)
        self.convs = nn.ModuleList([
            GATConv(self.gnn_hidden_size if i == 0 else self.gnn_hidden_size * self.gnn_heads,
                    self.gnn_hidden_size,
                    heads=self.gnn_heads,
                    dropout=config['transformer_dropout_rate'])
            for i in range(self.gnn_layers)
        ])

        # Readout layer
        self.readout = global_mean_pool

    def forward(self, v):
        """
        Input:
            v: Tuple (node_features, edge_index, batch_indices)
               - node_features: Amino acid IDs (from sequence)
               - edge_index: Graph connectivity (precomputed)
               - batch_indices: For graph batching
        """
        x, edge_index, batch = v[0].long(), v[1], v[2]

        # Node embeddings
        x = self.node_emb(x)

        # GNN message passing
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        # Graph-level readout
        graph_embedding = self.readout(x, batch)
        return graph_embedding

from torch_geometric.data import Data
import numpy as np

def protein_to_graph(sequence, pdb_structure=None):
    """
    Convert protein sequence to graph.
    If PDB structure is available, use spatial distances.
    Otherwise, use sequence proximity + predicted contacts.
    """
    # Example: Simple sequence-based graph (k-nearest residues)
    num_residues = len(sequence)
    edge_index = []
    # Standard 20 amino acids + special tokens
    aa_to_idx = {
        'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
        'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
        'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
        'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
        'X': 20,  # Unknown
        'B': 20,  # Aspartic acid or Asparagine (ambiguous)
        'Z': 20,  # Glutamic acid or Glutamine (ambiguous)
        '-': 21,  # Gap
        '<cls>': 22, '<pad>': 23, '<eos>': 24, '<mask>': 25
    }

    # Add sequential edges
    for i in range(num_residues - 1):
        edge_index.append([i, i+1])
        edge_index.append([i+1, i])

    # Add non-local edges (example: every 5th residue)
    for i in range(num_residues):
        for j in range(i+5, min(i+15, num_residues), 5):
            edge_index.append([i, j])
            edge_index.append([j, i])

    edge_index = torch.tensor(edge_index).t().contiguous()

    # Node features: amino acid indices
    x = torch.tensor([aa_to_idx[aa] for aa in sequence], dtype=torch.long)

    return Data(x=x, edge_index=edge_index)
from torch.utils import data
import sys
sys.path.append("../../DeepPurpose-master")
from DeepPurpose.utils import data_process
from tdc.multi_pred import DTI
from torch_geometric.data import Batch

def custom_protein_featurizer(sequence):
    graph = protein_to_graph(sequence)
    return (graph.x, graph.edge_index, torch.zeros(len(graph.x), dtype=torch.long))

# Example usage
dti_data = DTI(name = 'DAVIS')
all_data = dti_data.get_data()
df=all_data
#processed_data = df_data_preprocess(all_data)
#['Drug_ID', 'Drug', 'Target_ID', 'Target', 'Y']
X_drug, X_target, y = all_data['Drug'],all_data['Target'],all_data['Y']
X_target = [custom_protein_featurizer(seq) for seq in X_target]
config = {
    'input_dim_protein': 25,  # 20 AA + special tokens
    'gnn_hidden_size': 128,
    'gnn_layers': 3,
    'gnn_heads': 4,
    'threshold_distance': 8.0,
    'graph_type': 'distance',
    'transformer_dropout_rate':0.4
}

model = ProteinGraphEncoder(encoding='protein', **config)




model = model.to('cuda' if torch.cuda.is_available() else 'cpu')  # GPU if available

# 3. Prepare sample protein sequences (replace with your real data)
sample_sequences = ["MKKFFDSRRE", "GGSGLGSGSSGGGG"]  # Example sequences

# 4. Convert sequences to graphs
def prepare_batch(sequences):
    """Convert list of sequences to batched graph"""
    graph_list = []
    for seq in sequences:
        # Use the protein_to_graph function we defined earlier
        graph = protein_to_graph(seq)
        graph_list.append(graph)

    # Create a batch from individual graphs
    return Batch.from_data_list(graph_list)

# 5. Process through the model
batch = prepare_batch(sample_sequences)
batch = batch.to(model.device)  # Move to same device as model

# Forward pass
with torch.no_grad():  # Remove this for training
    embeddings = model((
        batch.x,          # Node features
        batch.edge_index, # Graph connectivity
        batch.batch       # Batch indices
    ))

print(f"Output embeddings shape: {embeddings.shape}")  # Should be [num_graphs, gnn_hidden_size]
