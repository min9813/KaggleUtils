import torch
import torch.nn as nn


class SimpleDAE(nn.Module):
    """
    get output of last layer as learned representation

    """

    def __init__(self, input_dim, layer_units, activations, lrelu_slope=0.1):
        if len(layer_units) != len(activations):
            raise ValueError(
                "length of 'layer_units' must equal to 'activations'")
        super(SimpleDAE, self).__init__()
        activation_dict = {
            "relu": nn.ReLU(),
            "lrelu": nn.LeakyReLU(lrelu_slope),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid()
        }
        layer = [nn.Linear(input_dim, layer_units[0])]
        layer.append(nn.Dropout(0.1))
        layer.append(activation_dict[activations[0]])
        self.first = nn.Sequential(*layer)
        layer = []

        if len(layer_units) > 1:
            for layer_id, layer_unit in enumerate(layer_units[1:]):
                layer.append(nn.Linear(layer_units[layer_id], layer_unit))
                layer.append(nn.Dropout(0.1))
                layer.append(activation_dict[activations[layer_id+1]])
                self.add_module(f"hidden_{layer_id}", nn.Sequential(*layer))
                layer = []

        self.output_layer = nn.Linear(layer_units[-1], input_dim)

    def forward(self, x):
        h = x
        for module in self.children():
            h = module(h)

        return h


class SharedDAE(nn.Module):
    """
    get output of last layer as learned representation

    """

    def __init__(self, layer_units, activations, input_dim=1, lrelu_slope=0.1):
        if len(layer_units) != len(activations):
            raise ValueError(
                "length of 'layer_units' must equal to 'activations'")
        super(SimpleDAE, self).__init__()
        activation_dict = {
            "relu": nn.ReLU(),
            "lrelu": nn.LeakyReLU(lrelu_slope),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid()
        }
        layer = [nn.Linear(input_dim, layer_units[0])]
        layer.append(activation_dict[activations[0]])

        if len(layer_units) > 1:
            for layer_id, layer_unit in enumerate(layer_units[1:]):
                layer.append(nn.Linear(layer_units[layer_id], layer_unit))
                layer.append(activation_dict[activations[layer_id+1]])

        self.learning_layer = nn.Sequential(*layer)

        self.output_layer = nn.Linear(layer_units[-1]*input_dim, input_dim)

    def forward(self, x):
        batchsize, feat_num = x.size()
        h = x.view(batchsize, feat_num, 1)
        h = self.learning_layer(x)
        h = h.view(batchsize, -1)

        return self.output_layer(h)
