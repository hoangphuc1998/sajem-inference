import torch.nn as nn
class MLP(nn.Module):
    '''
    Neural network with custom hidden layers
    '''
    def __init__(self, input_dim, output_dim, hidden_units, hidden_activation='relu', output_activation='relu', use_dropout = False, use_batchnorm=False):
        super().__init__()
        self.network = nn.Sequential()
        hidden_units = [input_dim] + hidden_units
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        
        for i in range(len(hidden_units) - 1):
            self.network.add_module("dense_" + str(i), nn.Linear(hidden_units[i], hidden_units[i+1]))
            # Hidden activation
            if hidden_activation == 'relu':
              self.network.add_module("activation_" + str(i), nn.ReLU())
            elif hidden_activation == 'sigmoid':
              self.network.add_module("activation_" + str(i), nn.Sigmoid())
            elif hidden_activation == 'tanh':
              self.network.add_module("activation_" + str(i), nn.Tanh())
            elif hidden_activation == 'lrelu':
              self.network.add_module("activation_" + str(i), nn.LeakyReLU())
            elif hidden_activation == 'prelu':
              self.network.add_module("activation_" + str(i), nn.PReLU())
            # Batchnorm on hidden layers
            if self.use_batchnorm:
              self.network.add_module("batchnorm_" + str(i), nn.BatchNorm1d(hidden_units[i+1]))
        
        # Dropout with 20% probability
        if self.use_dropout:
          self.network.add_module("dropout", nn.Dropout(0.2))

        self.network.add_module("output", nn.Linear(hidden_units[-1], output_dim))
        # Output activation
        if output_activation == 'relu':
          self.network.add_module("activation_out", nn.ReLU())
        elif output_activation == 'sigmoid':
          self.network.add_module("activation_out", nn.Sigmoid())
        elif output_activation == 'tanh':
          self.network.add_module("activation_out", nn.Tanh())

    def forward(self, x):
        return self.network(x)