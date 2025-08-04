import torch
import torch.nn as nn
import torch.nn.functional as F

class PGNNLayer(nn.Module):
    def __init__(self, in_features, out_features, num_anchors=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_anchors = num_anchors

        # Weight matrices
        self.W = nn.Parameter(torch.Tensor(in_features, out_features))
        self.a = nn.Parameter(torch.Tensor(2 * out_features, 1))

        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, fatoms, agraph):
        """
        Args:
            fatoms: (batch_size, num_nodes, in_features)
            agraph: (batch_size, num_nodes, max_neighbors)
        Returns:
            (batch_size, num_nodes, out_features)
        """
        batch_size, num_nodes, _ = fatoms.shape

        # 1. Feature transformation
        h = torch.matmul(fatoms, self.W)  # (batch_size, num_nodes, out_features)

        # 2. Select random anchors per batch
        anchor_indices = torch.stack([
            torch.randperm(num_nodes, device=fatoms.device)[:self.num_anchors]
            for _ in range(batch_size)
        ])  # (batch_size, num_anchors)

        # 3. Gather anchor features
        anchor_features = torch.gather(
            h,
            1,
            anchor_indices.unsqueeze(-1).expand(-1, -1, self.out_features)
        )  # (batch_size, num_anchors, out_features)

        # 4. Compute attention scores
        h_self = h.unsqueeze(2)  # (batch_size, num_nodes, 1, out_features)
        h_anchors = anchor_features.unsqueeze(1)  # (batch_size, 1, num_anchors, out_features)

        a_input = torch.cat([h_self + h_anchors, h_self - h_anchors], dim=-1)
        e = torch.matmul(a_input, self.a).squeeze(-1)  # (batch_size, num_nodes, num_anchors)

        # 5. Create attention mask
        # Convert agraph to binary mask (1=connected, 0=not connected)
        # Assuming agraph contains neighbor indices (with padding)
        attention_mask = torch.zeros(batch_size, num_nodes, self.num_anchors,
                                   device=fatoms.device)

        # Create mapping from neighbor indices to anchor indices
        for b in range(batch_size):
            for i in range(num_nodes):
                neighbors = agraph[b, i]  # Get neighbors for this node
                # Find which anchors are in the neighbors
                in_anchors = torch.isin(anchor_indices[b], neighbors)
                attention_mask[b, i] = in_anchors.float()

        # 6. Apply masked attention
        e = e.masked_fill(attention_mask == 0, -1e9)
        attention = F.softmax(e, dim=-1)  # (batch_size, num_nodes, num_anchors)

        # 7. Aggregate features
        h_prime = torch.matmul(attention, anchor_features)  # (batch_size, num_nodes, out_features)

        return h_prime
# Initialize layer
pgnn = PGNNLayer(in_features=6, out_features=32)

# Process batch
batch_fatoms = torch.randn(32, 600, 6)  # 32 molecules, 600 atoms, 6 features
batch_agraph = torch.randint(-1, 600, (32, 600, 39))  # Adjacency info

output = pgnn(batch_fatoms, batch_agraph)  # (32, 600, 32)
