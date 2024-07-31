import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, JumpingKnowledge

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class GRUModel(nn.Module):
    """
    The GRU Model for trajectory prediction
    input_size: The number of expected features in the input x
    hidden_size: The number of features in the hidden state h
    num_layers: Number of recurrent layers
    num_classes: The vector size of the output
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.fc2 = nn.Linear(int(hidden_size/2), output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x, h0=None):
        """
        Forward pass
        x: Input data
        h0: Preivous hidden state, h0 is None at the beginning of the sequence
        """
        if h0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, hidden = self.gru(x, h0)  
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        return out, hidden


class CrossAttention(nn.Module):
    """
    Cross Attention module
    d_model: The number of expected features in the input x
    num_heads: The number of heads in the multi-head attention
    """
    def __init__(self, d_model, num_heads=1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        # Linear layers to project q, k, v
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value):
        """
        Forward pass
        query: The query tensor, here it refers to a list of neighbors nodes (each node is a vector)
        key: The key tensor, here it referst to a list of all nodes (each node is a vector)
        value: The value tensor, the same as the key tensor
        """
        batch_size = query.size(0)

        Q = self.query_linear(query)
        K = self.key_linear(key)
        V = self.value_linear(value)
        
        # Reshape and transpose for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(attn_output)
        
        return output
    

class GDQN_Attention(nn.Module):
    """
    The Graph DQN model with Cross Attention
    in_channels: The number of expected features in the input x
    n_nodes: The number of nodes in the graph, this is for batch processing, not strictly necessary
    hidden_dim: The number of features in the hidden state of GCN
    max_n_neighbors: The maximum number of neighbors
    num_layers: Number of GCN layers
    """
    def __init__(self, in_channels=13, n_nodes=55, hidden_dim=32, max_n_neighbors=6, num_layers=5):
        super(GDQN_Attention, self).__init__()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.convs1 = GCNConv(in_channels, hidden_dim)
        self.convs2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * max_n_neighbors, max_n_neighbors)
        self.selu = nn.SELU()
        self.cross_attn = CrossAttention(hidden_dim, 1)
        self.max_n_neighbors = max_n_neighbors

    def forward(self, data, shuffle_indices=None):
        x, edge_index = data.x, data.edge_index
        batch_neighbor_mask = x[:, -3].reshape(-1, self.n_nodes)
        x = self.convs1(x, edge_index)
        x = self.selu(x)
        x = self.convs2(x, edge_index)
        x = self.selu(x)

        keys = x.reshape(-1, self.n_nodes, self.hidden_dim)
        values = x.reshape(-1, self.n_nodes, self.hidden_dim)
        batch_size = keys.size(0)
        querys = []
        for i in range(batch_size):
            neighbor_indices = torch.where(batch_neighbor_mask[i])
            query = keys[i][neighbor_indices]
            if query.size(0) < self.max_n_neighbors:
                query = F.pad(query, (0, 0, 0, self.max_n_neighbors - query.size(0)), "constant", 0)
            querys.append(query)
        querys = torch.stack(querys)

        if batch_size == 1:
            shuffle_indices = shuffle_indices.unsqueeze(0)
        if shuffle_indices is not None:
            batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, shuffle_indices.size(1))
            shuffled_query = querys[batch_indices, shuffle_indices]
        else:
            shuffled_query = querys
        x = self.cross_attn(shuffled_query, keys, values)
        x = x.reshape(-1, self.hidden_dim * self.max_n_neighbors)
        x = self.selu(x)
        x = self.fc1(x)
        return x


class SAGE_GDQN_Attention(nn.Module):
    """
    The GraphSAGE DQN model with Cross Attention
    in_channels: The number of expected features in the input x
    n_nodes: The number of nodes in the graph, this is for batch processing, not strictly necessary
    hidden_dim: The number of features in the hidden state of GCN
    max_n_neighbors: The maximum number of neighbors
    num_layers: Number of GCN layers
    """
    def __init__(self, in_channels=13, n_nodes=55, hidden_dim=32, max_n_neighbors=6, num_layers=5):
        super(SAGE_GDQN_Attention, self).__init__()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.convs1 = SAGEConv(in_channels, hidden_dim)
        self.convs2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * max_n_neighbors, max_n_neighbors)
        self.selu = nn.SELU()
        self.cross_attn = CrossAttention(hidden_dim, 1)
        self.max_n_neighbors = max_n_neighbors

    def forward(self, data, shuffle_indices=None):
        x, edge_index = data.x, data.edge_index
        batch_neighbor_mask = x[:, -3].reshape(-1, self.n_nodes)
        x = self.convs1(x, edge_index)
        x = self.selu(x)
        x = self.convs2(x, edge_index)
        x = self.selu(x)

        keys = x.reshape(-1, self.n_nodes, self.hidden_dim)
        values = x.reshape(-1, self.n_nodes, self.hidden_dim)
        batch_size = keys.size(0)
        querys = []
        for i in range(batch_size):
            neighbor_indices = torch.where(batch_neighbor_mask[i])
            query = keys[i][neighbor_indices]
            if query.size(0) < self.max_n_neighbors:
                query = F.pad(query, (0, 0, 0, self.max_n_neighbors - query.size(0)), "constant", 0)
            querys.append(query)
        querys = torch.stack(querys)

        if batch_size == 1:
            shuffle_indices = shuffle_indices.unsqueeze(0)
        if shuffle_indices is not None:
            batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, shuffle_indices.size(1))
            shuffled_query = querys[batch_indices, shuffle_indices]
        else:
            shuffled_query = querys
        x = self.cross_attn(shuffled_query, keys, values)
        x = x.reshape(-1, self.hidden_dim * self.max_n_neighbors)
        x = self.selu(x)
        x = self.fc1(x)
        return x


class JK_SAGE_GDQN_Attention(nn.Module):
    """
    The GraphSAGE DQN model with Cross Attention and Jumping Knowledge
    in_channels: The number of expected features in the input x
    n_nodes: The number of nodes in the graph, this is for batch processing, not strictly necessary
    hidden_dim: The number of features in the hidden state of GCN
    max_n_neighbors: The maximum number of neighbors
    num_layers: Number of GCN layers
    """
    def __init__(self, in_channels=13, n_nodes=55, hidden_dim=32, max_n_neighbors=6, num_layers=5):
        super(JK_SAGE_GDQN_Attention, self).__init__()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.convs = nn.ModuleList([SAGEConv(in_channels if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.jk = JumpingKnowledge('max', num_layers=num_layers, channels=hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * max_n_neighbors, max_n_neighbors)
        self.selu = nn.SELU()
        self.cross_attn = CrossAttention(hidden_dim, 1)
        self.max_n_neighbors = max_n_neighbors

    def forward(self, data, shuffle_indices=None):
        x, edge_index = data.x, data.edge_index
        batch_neighbor_mask = x[:, -3].reshape(-1, self.n_nodes)
        xs = []
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.selu(x)
            xs.append(x)
        x = self.jk(xs)
        x = self.selu(x)

        keys = x.reshape(-1, self.n_nodes, self.hidden_dim)
        values = x.reshape(-1, self.n_nodes, self.hidden_dim)
        batch_size = keys.size(0)
        querys = []
        for i in range(batch_size):
            neighbor_indices = torch.where(batch_neighbor_mask[i])
            query = keys[i][neighbor_indices]
            if query.size(0) < self.max_n_neighbors:
                query = F.pad(query, (0, 0, 0, self.max_n_neighbors - query.size(0)), "constant", 0)
            querys.append(query)
        querys = torch.stack(querys)

        if batch_size == 1:
            shuffle_indices = shuffle_indices.unsqueeze(0)
        if shuffle_indices is not None:
            batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, shuffle_indices.size(1))
            shuffled_query = querys[batch_indices, shuffle_indices]
        else:
            shuffled_query = querys
        x = self.cross_attn(shuffled_query, keys, values)
        x = x.reshape(-1, self.hidden_dim * self.max_n_neighbors)
        x = self.selu(x)
        x = self.fc1(x)
        return x
    


class DQN(nn.Module):
    """
    The DQN model for routing reinforcement learning
    in_channels: The number of expected features in the input x
    n_nodes: The number of nodes in the graph
    hidden_dim: The number of features in the hidden state of GCN
    max_n_neighbors: The maximum number of neighbors
    """
    def __init__(self, in_channels=4, n_nodes=57, hidden_dim=64, max_n_neighbors=15):
        super(DQN, self).__init__()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.convs1 = GCNConv(in_channels, hidden_dim)
        self.convs2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(in_channels * n_nodes, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, n_nodes)
        self.fc5 = nn.Linear(n_nodes, max_n_neighbors)
        self.softmax = nn.Softmax(dim=1)
        self.selu = nn.SELU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.reshape(-1, self.n_nodes * 4)
        x = self.fc1(x)
        x = self.selu(x)
        x = self.fc2(x)
        x = self.selu(x)
        x = self.fc3(x)
        x = self.selu(x)
        x = self.fc4(x)
        x = self.selu(x)
        x = self.fc5(x)
        return x
    

class GDQN(nn.Module):
    """
    The Graph DQN model for routing reinforcement learning
    in_channels: The number of expected features in the input x
    n_nodes: The number of nodes in the graph
    hidden_dim: The number of features in the hidden state of GCN
    max_n_neighbors: The maximum number of neighbors
    """
    def __init__(self, in_channels=4, n_nodes=57, hidden_dim=64, dropout=0.1, max_n_neighbors=15):
        super(GDQN, self).__init__()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.convs1 = GCNConv(in_channels, hidden_dim)
        self.convs2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * n_nodes, n_nodes)
        self.fc2 = nn.Linear(n_nodes, max_n_neighbors)
        self.softmax = nn.Softmax(dim=1)
        self.selu = nn.SELU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.convs1(x, edge_index)
        x = self.selu(x)
        x = self.convs2(x, edge_index)
        x = self.selu(x)
        x = x.reshape(-1, self.n_nodes * self.hidden_dim)
        x = self.fc1(x)
        x = self.selu(x)
        x = self.fc2(x)
        return x