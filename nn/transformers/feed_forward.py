import torch
from torch import nn as nn

from labml_helpers.module import Module


class FeedForward(Module):

    def __init__(self,
                 d_model: int,   # d_model is number of features in token embedding
                 d_ff: int,      # d_ff is number of features in hidden layer (4 times d_model is common
                 dropout: float = 0.1,  # dropout probability ofr the hidden layer of the FFN
                 activation: nn.Module = nn.ReLU(),  # activation function
                 is_gated: bool = False,    # specifies whether the hidden layer is gated
                 bias1: bool = True,    # specifies whether the first fully connected layer should have a learnable bias
                 bias2: bool = True,    # specifies whether the second fully connected layer should have a learnable bias
                 bias_gate: bool = True # specified whether the fully connected layer for the gate should have a learnable bias
                ):
        super().__init__()

        # Layer one parameterized by weight W1 and bias b1
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)

        # Layer two parameterized by weight W2 and bias b2
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)

        # Hidden layer dropout
        self.dropout = nn.Dropout(dropout)

        # Activation function
        self.activation = activation

        # Whether there is a gate
        self.is_gated = is_gated
        if is_gated:
            # If there is a gate the linear layer to transform inputs to be multiplied by the gate, parameterized by
            # weight V and bias bias_gate
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: torch.Tensor):
        # f(x * W1 + b1) and then apply activation function
        g = self.activation(self.layer1(x))

        # If there is a gate, f(x * W1 + b1) * g(x * V + b_gate)
        if self.is_gated:
            g = torch.sigmoid(self.linear_v(x)) * g
