import torch
import torch.nn as nn
import elmo_model as em

class Downstream_Model(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_layers, num_classes, dropout):
        super(Downstream_Model, self).__init__()

        self.embedding_matrix = embedding_matrix
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout

        self.elmo = em.ELMO(embedding_matrix, hidden_dim, num_layers, dropout, 'best_elmo_model.pth')

        for param in self.elmo.parameters():
            param.requires_grad = False

        self.delta = nn.Parameter(torch.randn(1, 3))
        self.linear = nn.Sequential(
            nn.Linear(2 * hidden_dim, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

    def forward(self, forward_input, backward_input, lengths):
        _, forward_h_n, forward_c_n, input = self.elmo(forward_input, lengths, 1)
        _, backward_h_n, backward_c_n, _ = self.elmo(backward_input, lengths, 0)

        input = torch.mean(input, dim=1)

        hidden = torch.cat([forward_h_n.permute(1, 0, 2), backward_h_n.permute(1, 0, 2)], dim=2)
        cell = torch.cat([forward_c_n.permute(1, 0, 2), backward_c_n.permute(1, 0, 2)], dim=2)  
        mean = (hidden + cell) / 2
        
        input = torch.cat([input] * mean.shape[1], dim=1).unsqueeze(1)

        output = torch.cat([mean, input], dim=1)
        output = torch.matmul(self.delta / torch.sum(self.delta), output)

        output = output.squeeze(0)

        output = self.linear(output)
        return output