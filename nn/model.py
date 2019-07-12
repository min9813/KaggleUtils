import torch.nn as nn
import torch


class Net(nn.Module):

    def __init__(self, input_dim, output_dim, layer_units=[256, 256], slope=0.2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_units = layer_units
        self.slope = slope
        super(Net, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, layer_units[0]))
        layers.append(nn.BatchNorm1d(layer_units[0]))
        layers.append(nn.LeakyReLU(slope))
#         layers.append(nn.ReLU())
        prev_unit = layer_units[0]
        hidden_units = layer_units[1:]

        if len(hidden_units) > 0:

            for layer_unit in hidden_units:
                layers.append(nn.Linear(prev_unit, layer_unit))
                layers.append(nn.BatchNorm1d(layer_unit))
                layers.append(nn.LeakyReLU(slope))
#                 layers.append(nn.ReLU())
                prev_unit = layer_unit

        layers.append(nn.Linear(prev_unit, output_dim))
#         layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)


#         self.dropout = nn.Dropout()
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.layers(x)

        return h


class Simple_NN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.75):
        super(Simple_NN, self).__init__()

        self.inpt_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(1, hidden_dim)
        self.relu = nn.ReLU()
        hidden = []
        hidden.append(nn.Linear(hidden_dim, hidden_dim))
#         hidden.append(nn.LeakyReLU(0.2))
#         curr_unit = hidden_dim //2
        curr_unit = hidden_dim
#         hidden.append(nn.Linear(curr_unit, curr_unit*2))
        hidden.append(nn.ReLU())
#         curr_unit = curr_unit *2
        self.fc_h = nn.Sequential(*hidden)
#         self.fc_h = nn.Linear(hidden_dim, hidden_dim/2)
        self.fc2 = nn.Linear(int(curr_unit*input_dim), output_dim)
        #self.fc3 = nn.Linear(int(hidden_dim/2*input_dim), int(hidden_dim/4))
        #self.fc4 = nn.Linear(int(hidden_dim/4*input_dim), int(hidden_dim/8))
        #self.fc5 = nn.Linear(int(hidden_dim/8*input_dim), 1)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)

        #self.bn2 = nn.BatchNorm1d(int(hidden_dim/2))
        #self.bn3 = nn.BatchNorm1d(int(hidden_dim/4))
        #self.bn4 = nn.BatchNorm1d(int(hidden_dim/8))

    def forward(self, x):
        b_size = x.size(0)
        x = x.view(-1, 1)
        y = self.fc1(x)
#         y = self.bn1(self.relu(y))
#         y = self.bn2(self.relu(self.fc_h(y)))
        y = self.fc1(x)
        y = self.relu(y)
        y = self.fc_h(y)
        y = y.view(b_size, -1)

        out = self.fc2(y)

        return out


class SelfAttnLinear(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, attn_dim=1):
        super(SelfAttnLinear, self).__init__()
        self.chanel_in = in_dim

        self.query_l = nn.Linear(in_dim, attn_dim)
        self.key_l = nn.Linear(in_dim, attn_dim)
        self.value_l = nn.Linear(in_dim, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batchsize, unit = x.size()
        h = x.view(batchsize, unit, 1)

        query = self.query_l(h)  # query shape:(B, H, D)
        key = self.key_l(h).permute(0, 2, 1)  # key shape:(B, D, H)
        attn = torch.bmm(query, key)
        attn = self.softmax(attn)
        # attn shape:(B, N, N)

        value = self.value_l(h).view(
            batchsize, unit, -1)  # value shape:(B, H, 1)
        value = torch.bmm(value, attn.permute(0, 2, 1))
        # value shape:(B, H, 1)
        value = value.view(batchsize, unit)

        value = self.gamma * value + x

        return value, attn


class AttentionNet(nn.Module):

    def __init__(self, input_dim, output_dim, attn_dim=1):
        self.attn = SelfAttnLinear(input_dim, attn_dim)
#         self.bn = nn.BatchNorm1d(input_dim)
        self.last_layer = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h, attn = self.attn(x)
        h = self.last_layer(h)

        return self.sigmoid(h), attn


class SimpleCNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.75):
        super(SimpleCNN, self).__init__()

        self.inpt_dim = input_dim
        self.hidden_dim = hidden_dim
        first = []
        curr_unit = hidden_dim[0]
        first.append(nn.Conv1d(1, curr_unit, kernel_size=1))
        first.append(nn.ReLU())
        first.append(nn.BatchNorm1d(curr_unit))
        self.first = nn.Sequential(*first)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        hidden_layer = []
        dim = input_dim
        kernel_size = 1
        for hidden in hidden_dim[1:]:
            hidden_layer.append(
                nn.Conv1d(curr_unit, hidden, kernel_size=kernel_size, stride=kernel_size))
            dim = dim // kernel_size
            hidden_layer.append(nn.ReLU())
            hidden_layer.append(nn.BatchNorm1d(hidden))
            curr_unit = hidden
        self.hidden = nn.Sequential(*hidden_layer)

        self.relu = nn.ReLU()
#         self.fc_h = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(int(max(1, dim)*curr_unit), output_dim)
        #self.fc3 = nn.Linear(int(hidden_dim/2*input_dim), int(hidden_dim/4))
        #self.fc4 = nn.Linear(int(hidden_dim/4*input_dim), int(hidden_dim/8))
        #self.fc5 = nn.Linear(int(hidden_dim/8*input_dim), 1)
        #self.bn1 = nn.BatchNorm1d(hidden_dim)
        #self.bn2 = nn.BatchNorm1d(int(hidden_dim/2))
        #self.bn3 = nn.BatchNorm1d(int(hidden_dim/4))
        #self.bn4 = nn.BatchNorm1d(int(hidden_dim/8))

    def forward(self, x):
        b_size = x.size(0)
        x = x.view(b_size, 1, -1)
        y = self.first(x)
        y = self.hidden(y)
        y = self.relu(y)
        y = y.view(b_size, -1)

        out = self.fc2(y)

        return out

# consider inner product of each feature.
# This is use explicitly product x and x


class CrossNet(nn.Module):

    def __init__(self, input_dim, output_dim, layer_units=[256], slope=0.2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.slope = slope
        self.layer_units = layer_units
        super(CrossNet, self).__init__()
        self.l1 = nn.Linear(input_dim, input_dim)
        prev_unit = input_dim
#         self.l2 = nn.Linear(input_dim, 1)

#         self.inter = nn.Linear(input_dim, input_dim)
#         hiddens = []
#         for layer_unit in layer_units:
#             hiddens.append(nn.Linear(prev_unit, layer_unit))
#             hiddens.append(nn.BatchNorm1d(layer_unit))
#             hiddens.append(nn.LeakyReLU(self.slope))
#             prev_unit = layer_unit

#         self.hiddens = nn.Sequential(*hiddens)

        self.l_last = nn.Linear(prev_unit*prev_unit+input_dim, output_dim)

        self.gamma = nn.Parameter(torch.ones(1))
#         self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, W = x.size()

#         h = h * x
        h = torch.bmm(x.view(batch, -1, 1), x.view(batch, 1, -1))  # (B, W, W)
        h = h.view(batch, -1)
        h = torch.cat([h, x], dim=1)
        h = self.l_last(h)

#         return self.sigmoid(h)
        return h
