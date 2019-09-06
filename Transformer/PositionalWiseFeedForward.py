import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(model_dim, ffn_dim)
        self.w_2 = nn.Linear(ffn_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x
        output = self.w_2(F.relu(self.w_1(output)))
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output
