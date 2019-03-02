import torch.nn as nn


class Net(nn.Module):

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(Net, self).__init__()
        self.linear1 = nn.Linear(input_dim, 1024)
#         self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 512)
#         self.linear4 = nn.Linear(512, 512)
#         self.linear5 = nn.Linear(512, 512)
        self.linear6 = nn.Linear(512, 256)
        self.linear7 = nn.Linear(256, 128)
#         self.linear8 = nn.Linear(128, 64)

#         self.linear_out = nn.Linear(64, output_dim)
        self.linear_out = nn.Linear(128, output_dim)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.dropout(self.leakyrelu(self.bn1(self.linear1(x))))
#         h = self.leakyrelu(self.bn1(self.linear2(h)))
        h = self.dropout(self.leakyrelu(self.bn2(self.linear3(h))))
#         h = self.dropout(self.leakyrelu(self.bn2(self.linear4(h))))
#         h = self.leakyrelu(self.bn2(self.linear5(h)))
#         h = self.leakyrelu(self.bn3(self.linear4(h)))
        h = self.dropout(self.leakyrelu(self.bn3(self.linear6(h))))
        h = self.leakyrelu(self.bn4(self.linear7(h)))

        return self.linear_out(h)
