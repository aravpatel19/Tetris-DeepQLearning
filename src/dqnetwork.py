import torch.nn as nn

class DQNetwork(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=1):
        super(DQNetwork, self).__init__()

        # first fully connected layer with relu activation
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # goes from input to a hidden layer
            nn.ReLU(inplace=True)  # activates neurons in this layer
        )

        # second fully connected layer, also with relu
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # another hidden layer
            nn.ReLU(inplace=True)  # activates neurons in this hidden layer
        )

        # final output layer with no activation (just gives the result)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # initialize the weights using xavier (glorot) initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)  # spread weights uniformly
                nn.init.constant_(module.bias, 0)  # set biases to zero

    def forward(self, inputs):
        x = self.layer1(inputs)  # pass input through the first layer
        x = self.layer2(x)       # then through the second layer
        output = self.output_layer(x)  # the last layer produces the predicted q-value

        return output
