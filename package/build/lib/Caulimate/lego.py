import torch
import torch.nn as nn
import torch.nn.functional as F

from .tools import check_tensor

class CustomMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(CustomMLP, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)  # No activation function on the output layer
        return x
    
    
class PartiallyPeriodicMLP(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, cos_len, periodic_ratio=0.2, cos_len_grad=False):
        """
            cos_len: period
        """
        super(PartiallyPeriodicMLP, self).__init__()

        # partially periodic mlp
        #self.cos_len = cos_len
        self.n_periodic_node = int(hid_dim * periodic_ratio)
        self.fc1 = nn.Linear(input_dim, hid_dim - self.n_periodic_node)
        if self.n_periodic_node != 0:
            self.fc1_ = nn.Linear(input_dim, self.n_periodic_node)
        
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        
        self.bias = nn.Parameter(torch.Tensor([4.0]))
        self.scale = nn.Parameter(torch.Tensor([8.0])) # 2
        self.cos_len_min_value = check_tensor([cos_len])
        self.cos_len_residual = nn.Parameter(torch.Tensor([1.0]))
        
        # init.uniform_(self.fc1.bias, -5000, 5000)

    def forward(self, t):
        x1 = self.sigmoid(self.fc1(t))
        if self.n_periodic_node != 0:
            x2 = self.fc1_(t)
            cos_len = self.cos_len_min_value# + nn.functional.softplus(self.cos_len_residual) 
            x2 = self.scale * torch.cos(2 * torch.pi * x2 / cos_len + self.bias)
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1

        x = self.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x
    
    

