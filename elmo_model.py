import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Forward_ELMO(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_layers, dropout):
        super(Forward_ELMO, self).__init__()

        self.embedding_matrix = embedding_matrix
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, embedding_matrix.shape[0])

    def forward(self, input_tensor, lengths):
        embedded = self.embedding(input_tensor)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=True)
        output, (h_n, c_n) = self.lstm(packed, None)
        padded, _ = pad_packed_sequence(output, batch_first=True)
        
        return padded, (h_n, c_n)
        
    
class Backward_ELMO(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_layers, dropout):
        super(Backward_ELMO, self).__init__()

        self.embedding_matrix = embedding_matrix
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, embedding_matrix.shape[0])

    def forward(self, input_tensor, lengths):
        embedded = self.embedding(input_tensor)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=True)
        output, (h_n, c_n) = self.lstm(packed, None)
        padded, _ = pad_packed_sequence(output, batch_first=True)
        
        return padded, (h_n, c_n)