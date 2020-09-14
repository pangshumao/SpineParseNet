import math
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from networks import graph
from matplotlib import pyplot as plt
# import pdb

class GraphConvolution(nn.Module):

    def __init__(self,in_features,out_features,bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features,out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias',None)

        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1./math.sqrt(self.weight(1))
        # self.weight.data.uniform_(-stdv,stdv)
        torch.nn.init.xavier_uniform_(self.weight)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv,stdv)

    def forward(self, input, adj=None,relu=False):
        support = torch.matmul(input,self.weight)
        # print(support.size(),adj.size())
        if adj is not None:
            output = torch.matmul(adj,support)
        else:
            output = support
        # print(output.size())
        if self.bias is not None:
            return output + self.bias
        else:
            if relu:
                return F.relu(output)
            else:
                return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Featuremaps_to_Graph(nn.Module):

    def __init__(self,input_channels,hidden_layers,nodes=7):
        super(Featuremaps_to_Graph, self).__init__()
        self.pre_fea = Parameter(torch.FloatTensor(input_channels,nodes))
        self.weight = Parameter(torch.FloatTensor(input_channels,hidden_layers))
        self.reset_parameters()

    def forward(self, input):
        assert input.dim() in [4, 5]
        if input.dim() == 4:
            n,c,h,w = input.size()
            # print('fea input',input.size())
            input1 = input.view(n,c,h*w)
            input1 = input1.transpose(1,2) # n x hw x c
            # print('fea input1', input1.size())
            ############## Feature maps to node ################
            fea_node = torch.matmul(input1,self.pre_fea) # n x hw x n_classes
            weight_node = torch.matmul(input1,self.weight) # n x hw x hidden_layer
            # softmax fea_node
            # fea_node = F.softmax(fea_node,dim=-1)
            # 2019-09-23
            fea_node = F.softmax(fea_node, dim=1)
            # print(fea_node.size(),weight_node.size())
            graph_node = F.relu(torch.matmul(fea_node.transpose(1,2),weight_node))
            return graph_node # n x n_class x hidden_layer
        elif input.dim() == 5:
            n, c, d, h, w = input.size()
            # print('fea input',input.size())
            input1 = input.view(n, c, d * h * w)
            input1 = input1.transpose(1, 2)  # n x dhw x c
            # print('fea input1', input1.size())
            ############## Feature maps to node ################
            fea_node = torch.matmul(input1, self.pre_fea)  # n x dhw x n_classes

            # written by Shumao Pang
            # fea_logit = fea_node.transpose(1, 2)  # n x n_classes x dhw
            # fea_logit = fea_logit.view(n, fea_node.size()[2], d, h, w)  # n x n_classes x d x h x w

            weight_node = torch.matmul(input1, self.weight)  # n x dhw x hidden_layer
            # softmax fea_node
            # fea_node = F.softmax(fea_node, dim=-1) # n x dhw x n_classes
            # 2019-09-23
            fea_node = F.softmax(fea_node, dim=1)  # n x dhw x n_classes

            # Witten by Shumao Pang
            # fea_sum = torch.sum(fea_node, dim=1).unsqueeze(1).expand(n, d * h * w, self.pre_fea.size()[-1])
            # fea_node = torch.div(fea_node, fea_sum)


            # print(fea_node.size(),weight_node.size())
            graph_node = F.relu(torch.matmul(fea_node.transpose(1, 2), weight_node)) # n x n_class x hidden_layer
            # return graph_node, fea_logit  # n x n_class x hidden_layer, n x n_classes x d x h x w
            return graph_node

    def reset_parameters(self):
        for ww in self.parameters():
            torch.nn.init.xavier_uniform_(ww)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv,stdv)

class My_Featuremaps_to_Graph(nn.Module):

    def __init__(self,input_channels, hidden_layers, nodes=7):
        super(My_Featuremaps_to_Graph, self).__init__()
        self.pre_fea = Parameter(torch.FloatTensor(input_channels,nodes))
        self.weight = Parameter(torch.FloatTensor(input_channels,hidden_layers))
        self.reset_parameters()

    def forward(self, input):
        assert input.dim() in [4, 5]
        if input.dim() == 4:
            n,c,h,w = input.size()
            # print('fea input',input.size())
            input1 = input.view(n, c, h*w)
            input1 = input1.transpose(1, 2) # n x hw x c
            # print('fea input1', input1.size())
            ############## Feature maps to node ################
            fea_node = torch.matmul(input1, self.pre_fea) # n x hw x n_classes

            # written by Shumao Pang
            fea_logit = fea_node.transpose(1, 2)  # n x n_classes x dhw
            fea_logit = fea_logit.view(n, fea_node.size()[2], h, w)  # n x n_classes x h x w

            weight_node = torch.matmul(input1, self.weight) # n x hw x hidden_layer
            # softmax fea_node
            fea_node = F.softmax(fea_node, dim=-1)

            # Witten by Shumao Pang
            fea_sum = torch.sum(fea_node, dim=1).unsqueeze(1).expand(n, h * w, self.pre_fea.size()[-1])
            fea_node = torch.div(fea_node, fea_sum)

            # print(fea_node.size(),weight_node.size())
            graph_node = F.relu(torch.matmul(fea_node.transpose(1, 2),weight_node))
            return graph_node, fea_logit # n x n_class x hidden_layer
        elif input.dim() == 5:
            n, c, d, h, w = input.size()
            # print('fea input',input.size())
            input1 = input.view(n, c, d * h * w)
            input1 = input1.transpose(1, 2)  # n x dhw x c
            # print('fea input1', input1.size())
            ############## Feature maps to node ################
            fea_node = torch.matmul(input1, self.pre_fea)  # n x dhw x n_classes

            # written by Shumao Pang
            fea_logit = fea_node.transpose(1, 2)  # n x n_classes x dhw
            fea_logit = fea_logit.view(n, fea_node.size()[2], d, h, w)  # n x n_classes x d x h x w

            weight_node = torch.matmul(input1, self.weight)  # n x dhw x hidden_layer
            # softmax fea_node
            fea_node = F.softmax(fea_node, dim=-1) # n x dhw x n_classes

            # Witten by Shumao Pang
            fea_sum = torch.sum(fea_node, dim=1).unsqueeze(1).expand(n, d * h * w, self.pre_fea.size()[-1])
            fea_node = torch.div(fea_node, fea_sum)

            # print(fea_node.size(),weight_node.size())
            graph_node = F.relu(torch.matmul(fea_node.transpose(1, 2), weight_node)) # n x n_class x hidden_layer
            return graph_node, fea_logit  # n x n_class x hidden_layer, n x n_classes x d x h x w

    def reset_parameters(self):
        for ww in self.parameters():
            torch.nn.init.xavier_uniform_(ww)

class Featuremaps_to_Graph_transfer(nn.Module):

    def __init__(self,input_channels,hidden_layers,nodes=7, source_nodes=20):
        super(Featuremaps_to_Graph_transfer, self).__init__()
        self.pre_fea = Parameter(torch.FloatTensor(input_channels,nodes))
        self.weight = Parameter(torch.FloatTensor(input_channels,hidden_layers))
        self.pre_fea_transfer = nn.Sequential(*[nn.Linear(source_nodes, source_nodes),nn.LeakyReLU(True),
                                                nn.Linear(source_nodes, nodes), nn.LeakyReLU(True)])
        # self.reset_parameters()

    def forward(self, input, source_pre_fea):
        self.pre_fea.data = self.pre_fea_learn(source_pre_fea)
        n,c,h,w = input.size()
        # print('fea input',input.size())
        input1 = input.view(n,c,h*w)
        input1 = input1.transpose(1,2) # n x hw x c
        # print('fea input1', input1.size())
        ############## Feature maps to node ################
        fea_node = torch.matmul(input1,self.pre_fea) # n x hw x n_classes
        weight_node = torch.matmul(input1,self.weight) # n x hw x hidden_layer
        # softmax fea_node
        fea_node = F.softmax(fea_node,dim=-1)
        # print(fea_node.size(),weight_node.size())
        graph_node = F.relu(torch.matmul(fea_node.transpose(1,2),weight_node))
        return graph_node # n x n_class x hidden_layer

    def pre_fea_learn(self, input):
        pre_fea = self.pre_fea_transfer.forward(input.unsqueeze(0)).squeeze(0)
        return self.pre_fea.data + pre_fea

class Graph_to_Featuremaps(nn.Module):
    # this is a special version
    def __init__(self,input_channels,output_channels,hidden_layers,nodes=7):
        super(Graph_to_Featuremaps, self).__init__()
        self.node_fea = Parameter(torch.FloatTensor(input_channels+hidden_layers,1))
        self.weight = Parameter(torch.FloatTensor(hidden_layers,output_channels))
        # self.reset_parameters()

    def reset_parameters(self):
        for ww in self.parameters():
            torch.nn.init.xavier_uniform_(ww)

    def forward(self, input, res_feature):
        '''

        :param input: 1 x batch x nodes x hidden_layer
        :param res_feature: batch x channels x h x w
        :return:
        '''
        batchi,channeli,hi,wi = res_feature.size()
        # print(res_feature.size())
        # print(input.size())
        try:
            _,batch,nodes,hidden = input.size()
        except:
            # print(input.size())
            input = input.unsqueeze(0)
            _,batch, nodes, hidden = input.size()

        assert batch == batchi
        input1 = input.transpose(0,1).expand(batch,hi*wi,nodes,hidden)
        res_feature_after_view = res_feature.view(batch,channeli,hi*wi).transpose(1,2)
        res_feature_after_view1 = res_feature_after_view.unsqueeze(2).expand(batch,hi*wi,nodes,channeli)
        new_fea = torch.cat((res_feature_after_view1,input1),dim=3)

        # print(self.node_fea.size(),new_fea.size())
        new_node = torch.matmul(new_fea, self.node_fea) # batch x hw x nodes x 1
        new_weight = torch.matmul(input, self.weight)  # 1 x batch x node x channel
        new_node = new_node.view(batch, hi*wi, nodes)
        feature_out = torch.matmul(new_node,new_weight)
        # print(feature_out.size())
        feature_out = feature_out.transpose(2,3).contiguous().view(res_feature.size())
        return F.relu(feature_out)

class Graph_to_Featuremaps_savemem(nn.Module):
    # this is a special version for saving gpu memory. The process is same as Graph_to_Featuremaps.
    def __init__(self, input_channels, output_channels, hidden_layers, nodes=7):
        super(Graph_to_Featuremaps_savemem, self).__init__()
        self.node_fea_for_res = Parameter(torch.FloatTensor(input_channels, 1))
        self.node_fea_for_hidden = Parameter(torch.FloatTensor(hidden_layers, 1))
        self.weight = Parameter(torch.FloatTensor(hidden_layers,output_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for ww in self.parameters():
            torch.nn.init.xavier_uniform_(ww)

    def forward(self, input, res_feature):
        '''

        :param input: 1 x batch x nodes x hidden_layer
        :param res_feature: batch x channels x h x w or batch x channels x d x h x w
        :return:
        '''
        assert res_feature.dim() in [4, 5]
        if res_feature.dim() == 4:
            batchi, channeli, hi, wi = res_feature.size()

            # print(res_feature.size())
            # print(input.size())
            try:
                _,batch,nodes,hidden = input.size()
            except:
                # print(input.size())
                input = input.unsqueeze(0)
                _,batch, nodes, hidden = input.size()

            assert batch == batchi
            input1 = input.transpose(0,1).expand(batch,hi*wi,nodes,hidden) # batch x hi*wi x nodes x hidden
            res_feature_after_view = res_feature.view(batch,channeli,hi*wi).transpose(1,2) # batch x hi*wi x channeli
            res_feature_after_view1 = res_feature_after_view.unsqueeze(2).expand(batch,hi*wi,nodes,channeli) # batch x hi*wi x nodes x channeli
            # new_fea = torch.cat((res_feature_after_view1,input1),dim=3)
            ## sim
            new_node1 = torch.matmul(res_feature_after_view1, self.node_fea_for_res) # batch x hi*wi x nodes x 1
            new_node2 = torch.matmul(input1, self.node_fea_for_hidden) # batch x hi*wi x nodes x 1
            new_node = new_node1 + new_node2
            ## sim end
            # print(self.node_fea.size(),new_fea.size())
            # new_node = torch.matmul(new_fea, self.node_fea) # batch x hw x nodes x 1
            new_weight = torch.matmul(input, self.weight) # 1 x batch x nodes x channel
            new_node = new_node.view(batch, hi*wi, nodes) # batch x hi*wi x nodes
            feature_out = torch.matmul(new_node,new_weight) # 1 x batch x hi*wi x channel
            # print(feature_out.size())
            feature_out = feature_out.transpose(2,3).contiguous().view(res_feature.size()) # batch x channeli x di x hi x wi
            return F.relu(feature_out)

        elif res_feature.dim() == 5:
            batchi, channeli, di, hi, wi = res_feature.size()
            # print(res_feature.size())
            # print(input.size())
            try:
                _, batch, nodes, hidden = input.size()
            except:
                # print(input.size())
                input = input.unsqueeze(0)
                _, batch, nodes, hidden = input.size()

            assert batch == batchi
            input1 = input.transpose(0, 1).expand(batch, di * hi * wi, nodes, hidden)  # batch x di*hi*wi x nodes x hidden
            res_feature_after_view = res_feature.view(batch, channeli, di * hi * wi).transpose(1, 2)  # batch x di*hi*wi x channeli
            res_feature_after_view1 = res_feature_after_view.unsqueeze(2).expand(batch, di * hi * wi, nodes,
                                                                                 channeli)  # batch x di*hi*wi x nodes x channeli
            # new_fea = torch.cat((res_feature_after_view1,input1),dim=3)
            ## sim
            new_node1 = torch.matmul(res_feature_after_view1, self.node_fea_for_res)  # batch x di*hi*wi x nodes x 1
            new_node2 = torch.matmul(input1, self.node_fea_for_hidden)  # batch x di*hi*wi x nodes x 1
            new_node = new_node1 + new_node2

            ## sim end
            # print(self.node_fea.size(),new_fea.size())
            # new_node = torch.matmul(new_fea, self.node_fea) # batch x dhw x nodes x 1
            new_weight = torch.matmul(input, self.weight)  # 1 x batch x node x channel
            new_node = new_node.view(batch, di * hi * wi, nodes)  # batch x di*hi*wi x nodes

            # 2019-09-23
            new_node = F.softmax(new_node, dim=-1)


            # f3 = new_node1.view(batch, di * hi * wi, nodes).transpose(1, 2) # batch x  nodes x di*hi*wi
            # f3 = f3.view(batch, nodes, di, hi, wi)
            # f3_cpu = f3.to('cpu').numpy()
            #
            # g2 = new_node2.view(batch, di * hi * wi, nodes).transpose(1, 2)  # batch x  nodes x di*hi*wi
            # g2 = g2.view(batch, nodes, di, hi, wi)
            # g2_cpu = g2.to('cpu').numpy()

            # f = new_node.transpose(1,2)
            # f = f.view(batch, nodes, di, hi, wi)
            # f_cpu = f.to('cpu').numpy()

            # plt.subplot(421)
            # plt.imshow(f_cpu[0, 1, 7, :, :])
            # plt.subplot(422)
            # plt.imshow(g2_cpu[0, 1, 7, :, :])
            #
            # plt.subplot(423)
            # plt.imshow(f_cpu[0, 2, 7, :, :])
            # plt.subplot(424)
            # plt.imshow(g2_cpu[0, 2, 7, :, :])
            #
            # plt.subplot(425)
            # plt.imshow(f_cpu[0, 3, 7, :, :])
            # plt.subplot(426)
            # plt.imshow(g2_cpu[0, 3, 7, :, :])
            #
            # plt.subplot(427)
            # plt.imshow(f_cpu[0, 4, 7, :, :])
            # plt.subplot(428)
            # plt.imshow(g2_cpu[0, 4, 7, :, :])
            #
            # plt.show()


            feature_out = torch.matmul(new_node, new_weight)  # 1 x batch x di*hi*wi x channel
            # print(feature_out.size())
            feature_out = feature_out.transpose(2, 3).contiguous().view(res_feature.size()) # batch x channeli x di x hi x wi
            feature_out = F.relu(feature_out)
            #
            # feature_out_cpu = feature_out.to('cpu')
            # plt.subplot(411)
            # plt.imshow(feature_out_cpu[0, 1, 7, :, :])
            # plt.subplot(412)
            # plt.imshow(feature_out_cpu[0, 2, 7, :, :])
            #
            # plt.subplot(413)
            # plt.imshow(feature_out_cpu[0, 3, 7, :, :])
            # plt.subplot(414)
            # plt.imshow(feature_out_cpu[0, 4, 7, :, :])
            # plt.show()

            return feature_out

class My_Graph_to_Featuremaps(nn.Module):
    def __init__(self, hidden_layers, output_channels, dimension=3):
        super(My_Graph_to_Featuremaps, self).__init__()
        self.hidden_layers = hidden_layers
        self.output_channels = output_channels
        if dimension == 3:
            self.conv = nn.Conv3d(hidden_layers, self.output_channels, 1, bias=False)
            self.bn = nn.BatchNorm3d(output_channels)
        else:
            self.conv = nn.Conv2d(hidden_layers, self.output_channels, 1, bias=False)
            self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(True)

    def forward(self, graph, fea_logit):
        '''
        :param graph: batch x nodes x hidden_layers
        :param fea_logit: batch x nodes x d x h x w
        :return: fea_map: batch x output_channels x d x h x w
        '''
        assert fea_logit.dim() in [4, 5]
        if fea_logit.dim() == 4:
            fea_prob = F.softmax(fea_logit, dim=1)  # batch x nodes x h x w
            batch, nodes, h, w = fea_prob.size()
            fea_prob = fea_prob.view(batch, nodes, h * w)  # batch x nodes x hw
            fea_prob = fea_prob.transpose(1, 2)  # batch x hw x nodes
            fea_map = torch.matmul(fea_prob, graph)  # batch x hw x hidden_layer
            fea_map = fea_map.transpose(1, 2)  # batch x hidden_layers x dhw
            fea_map = fea_map.view(batch, self.hidden_layers, h, w)  # batch x hidden_layers x h x w
            fea_map = self.conv(fea_map)  # batch x output_channels x h x w
            fea_map = self.bn(fea_map)
            fea_map = self.relu(fea_map)
            return fea_map
        else:
            fea_prob = F.softmax(fea_logit, dim=1) # batch x nodes x d x h x w
            batch, nodes, d, h, w = fea_prob.size()
            fea_prob = fea_prob.view(batch, nodes, d * h * w) # batch x nodes x dhw
            fea_prob = fea_prob.transpose(1, 2) # batch x dhw x nodes
            fea_map = torch.matmul(fea_prob, graph) # batch x dhw x hidden_layer
            fea_map = fea_map.transpose(1, 2) # batch x hidden_layers x dhw
            fea_map = fea_map.view(batch, self.hidden_layers, d, h, w) # batch x hidden_layers x d x h x w
            fea_map = self.conv(fea_map) # batch x output_channels x d x h x w
            fea_map = self.bn(fea_map)
            fea_map = self.relu(fea_map)
            return fea_map

class My_Graph_to_Featuremaps2(nn.Module):
    # this is a special version for saving gpu memory. The process is same as Graph_to_Featuremaps.
    def __init__(self, input_channels, output_channels, hidden_layers, nodes=7):
        super(My_Graph_to_Featuremaps2, self).__init__()
        # self.node_fea_for_res = Parameter(torch.FloatTensor(input_channels, 1))
        self.node_fea_for_hidden = Parameter(torch.FloatTensor(hidden_layers, 1))
        self.weight = Parameter(torch.FloatTensor(hidden_layers,output_channels))
        self.conv = nn.Conv3d(input_channels, nodes, kernel_size=3, padding=1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        for ww in self.parameters():
            torch.nn.init.xavier_uniform_(ww)

    def forward(self, input, res_feature):
        '''

        :param input: 1 x batch x nodes x hidden_layer
        :param res_feature: batch x channels x d x h x w
        :return:
        '''

        batchi, channeli, di, hi, wi = res_feature.size()
        # print(res_feature.size())
        # print(input.size())
        try:
            _, batch, nodes, hidden = input.size()
        except:
            # print(input.size())
            input = input.unsqueeze(0)
            _, batch, nodes, hidden = input.size()

        assert batch == batchi
        input1 = input.transpose(0, 1).expand(batch, di * hi * wi, nodes, hidden)  # batch x di*hi*wi x nodes x hidden
        new_weight = torch.matmul(input, self.weight)  # 1 x batch x node x channel

        logit = self.conv(res_feature) # batch x nodes x di x hi x wi

        new_node1 = logit.view(batch, nodes, di * hi * wi).transpose(1, 2)  # batch x di*hi*wi x nodes

        new_node2 = torch.matmul(input1, self.node_fea_for_hidden)  # batch x di*hi*wi x nodes x 1
        new_node2 = new_node2.view(batch, di * hi * wi, nodes) # batch x di*hi*wi x nodes
        new_node = new_node1 + new_node2

        # 2019-09-23
        new_node = F.softmax(new_node, dim=-1)

        feature_out = torch.matmul(new_node, new_weight)  # 1 x batch x di*hi*wi x channel
        # print(feature_out.size())
        feature_out = feature_out.transpose(2, 3).squeeze().view(res_feature.size()) # batch x channel x di x hi x wi
        feature_out = F.relu(feature_out)


        f = new_node2.transpose(1,2)
        f = f.view(batch, nodes, di, hi, wi)
        f_cpu = f.to('cpu').numpy()

        plt.subplot(421)
        plt.imshow(f_cpu[0, 1, 7, :, :])
        plt.subplot(422)
        plt.imshow(f_cpu[0, 2, 7, :, :])

        plt.subplot(423)
        plt.imshow(f_cpu[0, 3, 7, :, :])
        plt.subplot(424)
        plt.imshow(f_cpu[0, 4, 7, :, :])

        plt.subplot(425)
        plt.imshow(f_cpu[0, 5, 7, :, :])
        plt.subplot(426)
        plt.imshow(f_cpu[0, 6, 7, :, :])

        plt.subplot(427)
        plt.imshow(f_cpu[0, 7, 7, :, :])
        plt.subplot(428)
        plt.imshow(f_cpu[0, 8, 7, :, :])

        plt.show()


        # feature_out_cpu = feature_out.to('cpu')
        # plt.subplot(411)
        # plt.imshow(feature_out_cpu[0, 1, 7, :, :])
        # plt.subplot(412)
        # plt.imshow(feature_out_cpu[0, 2, 7, :, :])
        #
        # plt.subplot(413)
        # plt.imshow(feature_out_cpu[0, 3, 7, :, :])
        # plt.subplot(414)
        # plt.imshow(feature_out_cpu[0, 4, 7, :, :])
        # plt.show()

        return feature_out

class My_Graph_to_Featuremaps3(nn.Module):
    # this is a special version for saving gpu memory. The process is same as Graph_to_Featuremaps.
    def __init__(self, input_channels, output_channels, hidden_layers, nodes=7):
        super(My_Graph_to_Featuremaps3, self).__init__()
        self.node_fea_for_res = Parameter(torch.FloatTensor(input_channels, nodes))
        self.node_fea_for_hidden = Parameter(torch.FloatTensor(hidden_layers, 1))
        self.weight = Parameter(torch.FloatTensor(hidden_layers,output_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for ww in self.parameters():
            torch.nn.init.xavier_uniform_(ww)

    def forward(self, input, res_feature):
        '''

        :param input: 1 x batch x nodes x hidden_layer
        :param res_feature: batch x channels x d x h x w
        :return:
        '''

        batchi, channeli, di, hi, wi = res_feature.size()
        # print(res_feature.size())
        # print(input.size())
        try:
            _, batch, nodes, hidden = input.size()
        except:
            # print(input.size())
            input = input.unsqueeze(0)
            _, batch, nodes, hidden = input.size()

        assert batch == batchi
        input1 = input.transpose(0, 1).expand(batch, di * hi * wi, nodes, hidden)  # batch x di*hi*wi x nodes x hidden
        res_feature_after_view = res_feature.view(batch, channeli, di * hi * wi).transpose(1, 2)  # batch x di*hi*wi x channeli
        res_feature_after_view1 = res_feature_after_view.unsqueeze(2).expand(batch, di * hi * wi, nodes,
                                                                             channeli)  # batch x di*hi*wi x nodes x channeli

        new_node1 = torch.matmul(res_feature_after_view, self.node_fea_for_res)  # batch x di*hi*wi x nodes
        new_node2 = torch.matmul(input1, self.node_fea_for_hidden)  # batch x di*hi*wi x nodes x 1
        new_node2 = new_node2.view(batch, di*hi*wi, nodes) # batch x di*hi*wi x nodes
        new_node = new_node1 + new_node2

        new_weight = torch.matmul(input, self.weight)  # 1 x batch x node x channel

        # 2019-09-23
        new_node = F.softmax(new_node, dim=-1)


        # f3 = new_node1.view(batch, di * hi * wi, nodes).transpose(1, 2) # batch x  nodes x di*hi*wi
        # f3 = f3.view(batch, nodes, di, hi, wi)
        # f3_cpu = f3.to('cpu').numpy()
        #
        # g2 = new_node2.view(batch, di * hi * wi, nodes).transpose(1, 2)  # batch x  nodes x di*hi*wi
        # g2 = g2.view(batch, nodes, di, hi, wi)
        # g2_cpu = g2.to('cpu').numpy()
        #
        # f = new_node.transpose(1,2)
        # f = f.view(batch, nodes, di, hi, wi)
        # f_cpu = f.to('cpu').numpy()

        # plt.subplot(421)
        # plt.imshow(f_cpu[0, 1, 7, :, :])
        # plt.subplot(422)
        # plt.imshow(g2_cpu[0, 1, 7, :, :])
        #
        # plt.subplot(423)
        # plt.imshow(f_cpu[0, 2, 7, :, :])
        # plt.subplot(424)
        # plt.imshow(g2_cpu[0, 2, 7, :, :])
        #
        # plt.subplot(425)
        # plt.imshow(f_cpu[0, 3, 7, :, :])
        # plt.subplot(426)
        # plt.imshow(g2_cpu[0, 3, 7, :, :])
        #
        # plt.subplot(427)
        # plt.imshow(f_cpu[0, 4, 7, :, :])
        # plt.subplot(428)
        # plt.imshow(g2_cpu[0, 4, 7, :, :])
        #
        # plt.show()


        feature_out = torch.matmul(new_node, new_weight)  # 1 x batch x di*hi*wi x channel
        # print(feature_out.size())
        feature_out = feature_out.transpose(2, 3).contiguous().view(res_feature.size()) # batch x channeli x di x hi x wi
        feature_out = F.relu(feature_out)
        #
        # feature_out_cpu = feature_out.to('cpu')
        # plt.subplot(411)
        # plt.imshow(feature_out_cpu[0, 1, 7, :, :])
        # plt.subplot(412)
        # plt.imshow(feature_out_cpu[0, 2, 7, :, :])
        #
        # plt.subplot(413)
        # plt.imshow(feature_out_cpu[0, 3, 7, :, :])
        # plt.subplot(414)
        # plt.imshow(feature_out_cpu[0, 4, 7, :, :])
        # plt.show()

        return feature_out

class Graph_trans(nn.Module):

    def __init__(self,in_features,out_features,begin_nodes=7,end_nodes=2,bias=False,adj=None):
        super(Graph_trans, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features,out_features))
        if adj is not None:
            h,w = adj.size()
            assert (h == end_nodes) and (w == begin_nodes)
            self.adj = torch.autograd.Variable(adj,requires_grad=False)
        else:
            self.adj = Parameter(torch.FloatTensor(end_nodes,begin_nodes))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias',None)
        # self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1./math.sqrt(self.weight(1))
        # self.weight.data.uniform_(-stdv,stdv)
        torch.nn.init.xavier_uniform_(self.weight)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv,stdv)

    def forward(self, input, relu=False, adj_return=False, adj=None):
        support = torch.matmul(input,self.weight)
        # print(support.size(),self.adj.size())
        if adj is None:
            adj = self.adj
        adj1 = self.norm_trans_adj(adj)
        output = torch.matmul(adj1,support)
        if adj_return:
            output1 = F.normalize(output,p=2,dim=-1)
            self.adj_mat = torch.matmul(output1,output1.transpose(-2,-1))
        if self.bias is not None:
            return output + self.bias
        else:
            if relu:
                return F.relu(output)
            else:
                return output

    def get_adj_mat(self):
        adj = graph.normalize_adj_torch(F.relu(self.adj_mat))
        return adj

    def get_encode_adj(self):
        return self.adj

    def norm_trans_adj(self,adj):  # maybe can use softmax
        adj = F.relu(adj)
        r = F.softmax(adj,dim=-1)
        # print(adj.size())
        # row_sum = adj.sum(-1).unsqueeze(-1)
        # d_mat = row_sum.expand(adj.size())
        # r = torch.div(row_sum,d_mat)
        # r[torch.isnan(r)] = 0

        return r


if __name__ == '__main__':

    graph = torch.randn((7,128))
    pred = (torch.rand((7,7))*7).int()
    # a = en.forward(graph,pred)
    # print(a.size())