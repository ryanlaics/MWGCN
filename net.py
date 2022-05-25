from layer import *
# from AGCRN import *
import torch

class magnn(nn.Module):
    def __init__(self, gcn_depth, num_nodes, device, dropout=0.3, subgraph_size=20, node_dim=40, gnn_channels=32, scale_channels=16, end_channels=128, seq_length=12, layers=4, propalpha=0.05):
        super(magnn, self).__init__()
        self.num_nodes = num_nodes
        self.dropout = dropout

        self.device = device
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.scale_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()

        
        self.seq_length = seq_length
        self.layer_num = layers

        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, self.layer_num, device)
        
        self.kernel_set = [5, 4, 3, 2]

        self.scale_id = torch.autograd.Variable(torch.randn(self.layer_num, device=self.device), requires_grad=True)
        # self.scale_id = torch.arange(self.layer_num).to(device)
        self.lin1 = nn.Linear(self.layer_num, self.layer_num) 

        self.idx = torch.arange(self.num_nodes).to(device)
        self.scale_idx = torch.arange(self.num_nodes).to(device)

        length_set = []
        length_set.append(self.seq_length-self.kernel_set[0]+1)
        for i in range(1, self.layer_num):
            length_set.append(int((length_set[i-1]-self.kernel_set[i])/2))

        for i in range(self.layer_num):
            self.gconv1.append(mixprop(1, gnn_channels, gcn_depth, dropout, propalpha))
            self.gconv2.append(mixprop(1, gnn_channels, gcn_depth, dropout, propalpha))
            self.scale_convs.append(nn.Conv2d(in_channels=gnn_channels,
                                                    out_channels=1,
                                                    kernel_size=(1, length_set[i])))


        self.gated_fusion = gated_fusion(self.layer_num)
        self.end_conv_1 = nn.Conv2d(in_channels=1,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=1,
                                             kernel_size=(1,1),
                                             bias=True)
        self.fc1 = nn.Linear(130, int(130 // 2))
        self.fc2 = nn.Linear(int(130 // 2), int(130 // 4))
        self.fc3 = nn.Linear(int(130 // 4), 1)

    def forward(self, input):
        scale_input = []
        for i in range(len(input)):
            scale_input.append(torch.tensor(input[i]).float().to(self.device))
        print('scale_input',scale_input[0].size())
        print('scale_input1',scale_input[1].size())


        adj_matrix = self.gc(self.idx)
        out = []
        for i in range(self.layer_num):
            output = self.gconv1[i](scale_input[i], adj_matrix[i])+self.gconv2[i](scale_input[i], adj_matrix[i].transpose(1,0))
            scale_specific_output = self.scale_convs[i](output)
            out.append(scale_specific_output)
        out0 = torch.cat(out, dim=1)
        out1 = torch.stack(out, dim=1)
        outputs = self.gated_fusion(out0, out1)
        x = F.relu(outputs)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        xx = x.view(x.size(0), -1)
        xx = torch.relu(self.fc1(xx))
        xx = torch.relu(self.fc2(xx))
        xx = self.fc3(xx)
        xx = torch.sigmoid(xx.view(-1, 1))
        return xx, adj_matrix
