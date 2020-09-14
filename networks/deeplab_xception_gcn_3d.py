import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
from torchsummary import summary
from collections import OrderedDict
import os
from networks import gcn
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from networks import graph

class SeparableConv3d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv3d, self).__init__()

        self.conv1 = nn.Conv3d(inplanes, inplanes, kernel_size, stride, padding, dilation,
                               groups=inplanes, bias=bias)
        self.pointwise = nn.Conv3d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


def fixed_padding(inputs, kernel_size, rate):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv3d_aspp(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, padding=0):
        super(SeparableConv3d_aspp, self).__init__()

        self.depthwise = nn.Conv3d(inplanes, inplanes, kernel_size, stride, padding, dilation,
                                   groups=inplanes, bias=bias)
        self.depthwise_bn = nn.BatchNorm3d(inplanes)
        self.pointwise = nn.Conv3d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)
        self.pointwise_bn = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        #         x = fixed_padding(x, self.depthwise.kernel_size[0], rate=self.depthwise.dilation[0])
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.pointwise_bn(x)
        x = self.relu(x)
        return x

class Decoder_module(nn.Module):
    def __init__(self, inplanes, planes, rate=1):
        super(Decoder_module, self).__init__()
        self.atrous_convolution = SeparableConv3d_aspp(inplanes, planes, 3, stride=1, dilation=rate,padding=1)

    def forward(self, x):
        x = self.atrous_convolution(x)
        return x

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            raise RuntimeError()
        else:
            kernel_size = 3
            padding = rate
            self.atrous_convolution = SeparableConv3d_aspp(inplanes, planes, kernel_size, stride=1, dilation=rate,
                                                           padding=padding)

    def forward(self, x):
        x = self.atrous_convolution(x)
        return x

class ASPP_module_rate0(nn.Module):
    def __init__(self, inplanes, planes, rate=1):
        super(ASPP_module_rate0, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
            self.atrous_convolution = nn.Conv3d(inplanes, planes, kernel_size=kernel_size,
                                                stride=1, padding=padding, dilation=rate, bias=False)
            self.bn = nn.BatchNorm3d(planes, eps=1e-5, affine=True)
            self.relu = nn.ReLU()
        else:
            raise RuntimeError()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        return self.relu(x)

class SeparableConv3d_same(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, padding=0):
        super(SeparableConv3d_same, self).__init__()

        self.depthwise = nn.Conv3d(inplanes, inplanes, kernel_size, stride, padding, dilation,
                                   groups=inplanes, bias=bias)
        self.depthwise_bn = nn.BatchNorm3d(inplanes)
        self.pointwise = nn.Conv3d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)
        self.pointwise_bn = nn.BatchNorm3d(planes)

    def forward(self, x):
        x = fixed_padding(x, self.depthwise.kernel_size[0], rate=self.depthwise.dilation[0])
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = self.pointwise(x)
        x = self.pointwise_bn(x)
        return x

class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv3d(inplanes, planes, 1, stride=stride, bias=False)
            if is_last:
                self.skip = nn.Conv3d(inplanes, planes, 1, stride=1, bias=False)
            self.skipbn = nn.BatchNorm3d(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv3d_same(inplanes, planes, 3, stride=1, dilation=dilation))
#             rep.append(nn.BatchNorm3d(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv3d_same(filters, filters, 3, stride=1, dilation=dilation))
#             rep.append(nn.BatchNorm3d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv3d_same(inplanes, planes, 3, stride=1, dilation=dilation))
#             rep.append(nn.BatchNorm3d(planes))

        if not start_with_relu:
            rep = rep[1:]

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv3d_same(planes, planes, 3, stride=stride,dilation=dilation))

        if is_last:
            rep.append(self.relu)
            rep.append(SeparableConv3d_same(planes, planes, 3, stride=1,dilation=dilation))


        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        # print(x.size(),skip.size())
        x += skip

        return x

class Block2(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, start_with_relu=True, grow_first=True, is_last=False):
        super(Block2, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv3d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm3d(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv3d_same(inplanes, planes, 3, stride=1, dilation=dilation))
#             rep.append(nn.BatchNorm3d(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv3d_same(filters, filters, 3, stride=1, dilation=dilation))
#             rep.append(nn.BatchNorm3d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv3d_same(inplanes, planes, 3, stride=1, dilation=dilation))
#             rep.append(nn.BatchNorm3d(planes))

        if not start_with_relu:
            rep = rep[1:]

        if stride != 1:
            self.block2_lastconv = nn.Sequential(*[self.relu,SeparableConv3d_same(planes, planes, 3, stride=stride,dilation=dilation)])

        if is_last:
            rep.append(SeparableConv3d_same(planes, planes, 3, stride=1))


        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        low_middle = x.clone()
        x1 = x
        x1 = self.block2_lastconv(x1)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x1 += skip

        return x1,low_middle

class Xception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, inplanes=3, os=16, pretrained=False):
        super(Xception, self).__init__()

        if os == 16:
            entry_block3_stride = (1, 2, 2)
            middle_block_rate = 1
            exit_block_rates = (1, 2)
        elif os == 8:
            entry_block3_stride = 1
            middle_block_rate = 2
            exit_block_rates = (2, 4)
        else:
            raise NotImplementedError


        # Entry flow
        self.conv1 = nn.Conv3d(inplanes, 16, 3, stride=(1,2,2), padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(16, 32, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(32)

        self.block1 = Block(32, 64, reps=2, stride=(1,2,2), start_with_relu=False, grow_first=True)
        self.block2 = Block2(64, 128, reps=2, stride=(1,2,2), start_with_relu=True, grow_first=True)
        self.block3 = Block(128, 364, reps=2, stride=entry_block3_stride, start_with_relu=True, grow_first=True)

        # Middle flow
        # self.block4  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block5  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block6  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block7  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block8  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block9  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block10 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block11 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block12 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block13 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block14 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block15 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block16 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block17 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block18 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)
        # self.block19 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block20 = Block(364, 512, reps=2, stride=1, dilation=exit_block_rates[0],
                             start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv3d_aspp(512, 768, 3, stride=1, dilation=exit_block_rates[1],padding=exit_block_rates[1])
        # self.bn3 = nn.BatchNorm3d(1536)

        self.conv4 = SeparableConv3d_aspp(768, 768, 3, stride=1, dilation=exit_block_rates[1],padding=exit_block_rates[1])
        # self.bn4 = nn.BatchNorm3d(1536)

        self.conv5 = SeparableConv3d_aspp(768, 1024, 3, stride=1, dilation=exit_block_rates[1],padding=exit_block_rates[1])
        # self.bn5 = nn.BatchNorm3d(2048)

        # Init weights
        # self.__init_weight()

        # Load pretrained model
        if pretrained:
            self.__load_xception_pretrained()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print('conv1 ',x.size())
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        # print('block1',x.size())
        # low_level_feat = x
        x,low_level_feat = self.block2(x)
        # print('block2',x.size())
        x = self.block3(x)
        # print('xception block3 ',x.size())

        # Middle flow
        # x = self.block4(x)
        # x = self.block5(x)
        # x = self.block6(x)
        # x = self.block7(x)
        # x = self.block8(x)
        # x = self.block9(x)
        # x = self.block10(x)
        # x = self.block11(x)
        # x = self.block12(x)
        # x = self.block13(x)
        # x = self.block14(x)
        # x = self.block15(x)
        # x = self.block16(x)
        # x = self.block17(x)
        # x = self.block18(x)
        # x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.conv3(x)
        # x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        # x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        # x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_feat

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def __load_xception_pretrained(self):
        pretrain_dict = model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth')
        model_dict = {}
        state_dict = self.state_dict()

        for k, v in pretrain_dict.items():
            if k in state_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('block11'):
                    model_dict[k.replace('block11', 'block12')] = v
                    model_dict[k.replace('block11', 'block13')] = v
                    model_dict[k.replace('block11', 'block14')] = v
                    model_dict[k.replace('block11', 'block15')] = v
                    model_dict[k.replace('block11', 'block16')] = v
                    model_dict[k.replace('block11', 'block17')] = v
                    model_dict[k.replace('block11', 'block18')] = v
                    model_dict[k.replace('block11', 'block19')] = v
                elif k.startswith('conv3'):
                    model_dict[k] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

class DeepLabv3_plus_gcn_3d(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=21, os=16, pretrained=False, _print=True, final_sigmoid=False,
                 hidden_layers=128, device=None):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(DeepLabv3_plus_gcn_3d, self).__init__()
        self.device = device
        # Atrous Conv
        self.xception_features = Xception(nInputChannels, os, pretrained)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module_rate0(1024, 128, rate=rates[0])
        self.aspp2 = ASPP_module(1024, 128, rate=rates[1])
        self.aspp3 = ASPP_module(1024, 128, rate=rates[2])
        self.aspp4 = ASPP_module(1024, 128, rate=rates[3])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
                                             nn.Conv3d(1024, 128, 1, stride=1, bias=False),
                                             nn.BatchNorm3d(128),
                                             nn.ReLU()
                                             )

        self.concat_projection_conv1 = nn.Conv3d(640, 128, 1, bias=False)
        self.concat_projection_bn1 = nn.BatchNorm3d(128)

        # adopt [1x1, 48] for channel reduction.
        self.feature_projection_conv1 = nn.Conv3d(128, 24, 1, bias=False)
        self.feature_projection_bn1 = nn.BatchNorm3d(24)

        # self.decoder = nn.Sequential(Decoder_module(152, 128),
        #                              Decoder_module(128, 128)
        #                              )
        #
        # self.featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=128, hidden_layers=hidden_layers,
        #                                                    nodes=n_classes)
        # self.graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        # self.graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        # self.graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        #
        # self.graph_2_fea = gcn.Graph_to_Featuremaps_savemem(input_channels=128,
        #                                                     output_channels=128,
        #                                                     hidden_layers=hidden_layers,
        #                                                     nodes=n_classes)
        #
        # self.skip_conv = nn.Sequential(*[nn.Conv3d(128, 128, kernel_size=1),
        #                                         nn.ReLU(True)])
        #
        # self.semantic = nn.Conv3d(128, n_classes, kernel_size=1, stride=1)



        self.decoder = nn.Sequential(Decoder_module(152, 256),
                                     Decoder_module(256, 256)
                                     )

        self.featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=256, hidden_layers=hidden_layers,
                                                           nodes=n_classes)
        self.graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.graph_2_fea = gcn.Graph_to_Featuremaps_savemem(input_channels=256,
                                                            output_channels=256,
                                                            hidden_layers=hidden_layers,
                                                            nodes=n_classes)

        self.skip_conv = nn.Sequential(*[nn.Conv3d(256, 256, kernel_size=1),
                                         nn.ReLU(True)])

        self.semantic = nn.Conv3d(256, n_classes, kernel_size=1, stride=1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def get_graph(self):
        spine_adj = graph.preprocess_adj(graph.spine_graph)
        spine_adj_ = torch.from_numpy(spine_adj).float()
        spine_adj = spine_adj_.unsqueeze(0).to(self.device)
        return spine_adj

    def forward(self, input):
        adj = self.get_graph()
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='trilinear', align_corners=True)
        # x5 = F.upsample(x5, size=x4.size()[2:], mode='trilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = F.interpolate(x, size=low_level_features.size()[2:], mode='trilinear', align_corners=True)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)

        # feature map -> graph
        spine_graph, fea_logit = self.featuremap_2_graph(x)
        # graph convolution by 3 times
        spine_graph = self.graph_conv1.forward(spine_graph, adj=adj, relu=True)
        spine_graph = self.graph_conv2.forward(spine_graph, adj=adj, relu=True)
        spine_graph = self.graph_conv3.forward(spine_graph, adj=adj, relu=True)
        # graph -> feature map
        spine_graph = self.graph_2_fea.forward(spine_graph, x)

        x = self.skip_conv(x)
        x = x + spine_graph


        x = self.semantic(x)
        final_conv = F.interpolate(x, size=input.size()[2:], mode='trilinear', align_corners=True)
        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x = self.final_activation(final_conv)
            return x, final_conv
        else:
            return final_conv, fea_logit


    def freeze_bn(self):
        for m in self.xception_features.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()

    def freeze_totally_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()

    def freeze_aspp_bn(self):
        for m in self.aspp1.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()
        for m in self.aspp2.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()
        for m in self.aspp3.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()
        for m in self.aspp4.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()

    def learnable_parameters(self):
        layer_features_BN = []
        layer_features = []
        layer_aspp = []
        layer_projection  =[]
        layer_decoder = []
        layer_other = []
        model_para = list(self.named_parameters())
        for name,para in model_para:
            if 'xception' in name:
                if 'bn' in name or 'downsample.1.weight' in name or 'downsample.1.bias' in name:
                    layer_features_BN.append(para)
                else:
                    layer_features.append(para)
                    # print (name)
            elif 'aspp' in name:
                layer_aspp.append(para)
            elif 'projection' in name:
                layer_projection.append(para)
            elif 'decode' in name:
                layer_decoder.append(para)
            elif 'global' not in name:
                layer_other.append(para)
        return layer_features_BN,layer_features,layer_aspp,layer_projection,layer_decoder,layer_other

    def get_backbone_para(self):
        layer_features = []
        other_features = []
        model_para = list(self.named_parameters())
        for name, para in model_para:
            if 'xception' in name:
                layer_features.append(para)
            else:
                other_features.append(para)

        return layer_features, other_features

    def train_fixbn(self, mode=True, freeze_bn=True, freeze_bn_affine=False):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Returns:
            Module: self
        """
        super(DeepLabv3_plus_gcn_3d, self).train(mode)
        if freeze_bn:
            print("Freezing Mean/Var of BatchNorm3D.")
            if freeze_bn_affine:
                print("Freezing Weight/Bias of BatchNorm3D.")
        if freeze_bn:
            for m in self.xception_features.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    if freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
            # for m in self.aspp1.modules():
            #     if isinstance(m, nn.BatchNorm3d):
            #         m.eval()
            #         if freeze_bn_affine:
            #             m.weight.requires_grad = False
            #             m.bias.requires_grad = False
            # for m in self.aspp2.modules():
            #     if isinstance(m, nn.BatchNorm3d):
            #         m.eval()
            #         if freeze_bn_affine:
            #             m.weight.requires_grad = False
            #             m.bias.requires_grad = False
            # for m in self.aspp3.modules():
            #     if isinstance(m, nn.BatchNorm3d):
            #         m.eval()
            #         if freeze_bn_affine:
            #             m.weight.requires_grad = False
            #             m.bias.requires_grad = False
            # for m in self.aspp4.modules():
            #     if isinstance(m, nn.BatchNorm3d):
            #         m.eval()
            #         if freeze_bn_affine:
            #             m.weight.requires_grad = False
            #             m.bias.requires_grad = False
            # for m in self.global_avg_pool.modules():
            #     if isinstance(m, nn.BatchNorm3d):
            #         m.eval()
            #         if freeze_bn_affine:
            #             m.weight.requires_grad = False
            #             m.bias.requires_grad = False
            # for m in self.concat_projection_bn1.modules():
            #     if isinstance(m, nn.BatchNorm3d):
            #         m.eval()
            #         if freeze_bn_affine:
            #             m.weight.requires_grad = False
            #             m.bias.requires_grad = False
            # for m in self.feature_projection_bn1.modules():
            #     if isinstance(m, nn.BatchNorm3d):
            #         m.eval()
            #         if freeze_bn_affine:
            #             m.weight.requires_grad = False
            #             m.bias.requires_grad = False

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_state_dict_new(self, state_dict):
        own_state = self.state_dict()
        #for name inshop_cos own_state:
        #    print name
        new_state_dict = OrderedDict()
        for name, param in state_dict.items():
            name = name.replace('module.','')
            new_state_dict[name] = 0
            if name not in own_state:
                if 'num_batch' in name:
                    continue
                print ('unexpected key "{}" in state_dict'
                       .format(name))
                continue
                # if isinstance(param, own_state):
            if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                          ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                        name, own_state[name].size(), param.size()))
                continue # i add inshop_cos 2018/02/01
                # raise
                    # print 'copying %s' %name
                # if isinstance(param, own_state):
                # backwards compatibility for serialized parameters
            own_state[name].copy_(param)
            # print 'copying %s' %name

        missing = set(own_state.keys()) - set(new_state_dict.keys())
        if len(missing) > 0:
            print('missing keys in state_dict: "{}"'.format(missing))


def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.xception_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == "__main__":
    device = torch.device('cuda:3')
    image = torch.randn(2, 3, 20, 256, 512) * 255
    image = image.to(device)
    model = DeepLabv3_plus_gcn_3d(nInputChannels=3, n_classes=21, os=16, pretrained=False, _print=True)
    model = model.to(device)
    print(model)
    # summary(model, input_size=(3, 20, 256, 512))
    model.eval()

    with torch.no_grad():
        output = model.forward(image)
    print(output.size())
    # print(output)






