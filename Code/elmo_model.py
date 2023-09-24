import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ELMO(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_layers, dropout, filename=None):
        super(ELMO, self).__init__()

        self.embedding_matrix = embedding_matrix
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        self.forward_lstm = nn.LSTM(embedding_matrix.shape[1], hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.backward_lstm = nn.LSTM(embedding_matrix.shape[1], hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.forward_linear = nn.Linear(hidden_dim, embedding_matrix.shape[0])
        self.backward_linear = nn.Linear(hidden_dim, embedding_matrix.shape[0])

        if filename:
            self.load_state_dict(torch.load(filename), strict=False)

    def forward(self, input_tensor, lengths, forward):
        embedded = self.embedding(input_tensor)
        output = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        if forward:
            output, (h_n, c_n) = self.forward_lstm(output, None)
            output, _ = pad_packed_sequence(output, batch_first=True)
            output = self.forward_linear(output)
        else:
            output, (h_n, c_n) = self.backward_lstm(output, None)
            output, _ = pad_packed_sequence(output, batch_first=True)
            output = self.backward_linear(output)

        return output, h_n, c_n, embedded