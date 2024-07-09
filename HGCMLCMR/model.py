import torch
import numpy
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Parameter
from torchvision import models
from hypergraph_model import hypergraph_utils as hgut
from hypergraph_model.HGNN import HGNN
from util import *
from torch.nn import Linear
from collections import OrderedDict
import mxnet as mx
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn as nnn


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):  # 默认的权重参数初始化
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout=0, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.vgg = models.vgg19_bn(pretrained=True)
        self.vgg_features = self.vgg.features
        self.fc_features = nn.Sequential(*list(self.vgg.classifier.children())[:-2])

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.vgg_features(x).view(x.shape[0], -1)
        features = self.fc_features(features)
        return features


class TxtMLP(nn.Module):
    def __init__(self, code_len=300, txt_bow_len=1386, num_class=24):
        super(TxtMLP, self).__init__()
        self.fc1 = nn.Linear(txt_bow_len, 4096)
        self.fc2 = nn.Linear(4096, code_len)
        self.classifier = nn.Linear(code_len, num_class)

    def forward(self, x):
        feat = F.leaky_relu(self.fc1(x), 0.2)
        feat = F.leaky_relu(self.fc2(feat), 0.2)
        predict = self.classifier(feat)
        return feat, predict


class ImgNN(nn.Module):
    """Network to learn image representations"""

    def __init__(self, input_dim=4096, output_dim=1024):
        super(ImgNN, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.bn(x)
        out = F.relu(self.denseL1(x))
        return out


class TextNN(nn.Module):
    """Network to learn text representations"""

    def __init__(self, input_dim=1024, output_dim=1024):
        super(TextNN, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.bn(x)
        out = F.relu(self.denseL1(x))
        return out


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class ModalClassifier(nn.Module):
    """Network to discriminate modalities"""

    def __init__(self, input_dim=40):
        super(ModalClassifier, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.denseL1 = nn.Linear(input_dim, input_dim // 4)
        self.denseL2 = nn.Linear(input_dim // 4, input_dim // 16)
        self.denseL3 = nn.Linear(input_dim // 16, 2)

    def forward(self, x):
        x = ReverseLayerF.apply(x, 1.0)
        x = self.bn(x)
        out = F.relu(self.denseL1(x))
        out = F.relu(self.denseL2(out))
        out = self.denseL3(out)
        return out


class ImgDec(nn.Module):
    """Network to decode image representations"""

    def __init__(self, input_dim=1024, output_dim=4096, hidden_dim=2048):
        super(ImgDec, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.denseL1 = nn.Linear(input_dim, hidden_dim)
        self.denseL2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.bn(x)
        out = F.relu(self.denseL1(x))
        out = F.relu(self.denseL2(out))
        return out


class TextDec(nn.Module):
    """Network to decode image representations"""

    def __init__(self, input_dim=1024, output_dim=300, hidden_dim=512):
        super(TextDec, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.denseL1 = nn.Linear(input_dim, hidden_dim)
        self.denseL2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.bn(x)
        out = F.leaky_relu(self.denseL1(x), 0.2)
        out = F.leaky_relu(self.denseL2(out), 0.2)
        return out
class LocationAdaptiveLearner(nn.Module):
    """docstring for LocationAdaptiveLearner"""
    def __init__(self, in_channels, out_channels):
        super(LocationAdaptiveLearner, self).__init__()
        # self.nclass = nclass

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels,1, bias=True),

                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels,1, bias=True),

                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(out_channels, out_channels,1, bias=True)
                                   )

    def forward(self, x):
        # x:side5_w (N, 19*4, H, W)
        x = self.conv1(x) # (N, 19*4, H, W)
        x = self.conv2(x) # (N, 19*4, H, W)
        x = self.conv3(x) # (N, 19*4, H, W)
        x = x.view(x.size(0), -1, x.size(2), x.size(3)) # (N, 19, 4, H, W)
        return x
class DAF(nn.Module):
    '''
    直接相加 DirectAddFuse
    '''

    def __init__(self):
        super(DAF, self).__init__()

    def forward(self, x, residual):
        return x + residual

class iAFF(nn.Module):

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.local_att2 = nn.Sequential(
            # self.localadaptive(torch.tensor(channels)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        #xa = x + residual
        xl = self.local_att(x)
        xg = self.global_att(residual)
        xlg = xl + xg
        # wei = self.sigmoid(xlg)
        # xi = x * wei + residual * (1 - wei)
        # xl2 = self.local_att2(xi)
        # xg2 = self.global_att(xi)
        # xlg2 = xl2 + xg2
        # wei2 = self.sigmoid(xlg2)
        # xo = x * wei2 + residual * (1 - wei2)

        # global_feat_norm = torch.norm(xg, p=2, dim=1)  # 计算全局特征的L2范数
        # projection = torch.bmm(xg.squeeze(3), torch.flatten(
        #     xl, start_dim=2).transpose(1, 2))  # 计算全局特征与展平后的局部特征的投影
        # projection = torch.bmm(xg.squeeze(
        #     3).transpose(1, 2), projection).view(xl.size())  # 再次计算全局特征与投影的投影
        # projection = projection / \
        #              (global_feat_norm * global_feat_norm).view(-1, 1, 1, 1)  # 对投影进行归一化
        # orthogonal_comp = xl - projection  # 计算正交补
        # global_feat = xg  # 增加维度以适应拼接
        # xlg=global_feat+orthogonal_comp
        # xo = torch.cat([global_feat.expand(orthogonal_comp.size()), orthogonal_comp], dim=1)  # 在通道维度上拼接全局特征和正交补

        w = self.sigmoid(xlg)
        # w2 = self.sigmoid(xg)
        # w = (w1+w2)/2
        xo = x * w + 2*residual #* (1 - wei2)

        return xo

class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class NAFF(nn.Module):

    def __init__(self, channels=64, r=4):
        super(NAFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels * 2, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x, ds_y):
        ds_y =self.global_att(ds_y)
        xa = torch.cat((x, ds_y), dim=1)
        x_att = self.local_att(xa)
        x_att = 1.0 + torch.tanh(x_att)
        xo = torch.mul(x, x_att) + torch.mul(ds_y, 2.0-x_att)+x+ds_y

        return xo

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class HG(nn.Module):
    def __init__(self, img_input_dim=4096, text_input_dim=1024, minus_one_dim=1024, num_classes=10, in_channel=300, t=0,
                 adj_file=None, inp=None, GNN='GAT', n_layers=4):
        super(HG, self).__init__()
        self.img_net = ImgNN(img_input_dim, minus_one_dim)
        self.text_net = TextNN(text_input_dim, minus_one_dim)
        self.img2text_net = TextDec(minus_one_dim, text_input_dim)
        self.text2img_net = ImgDec(minus_one_dim, img_input_dim)
        self.img_md_net = ModalClassifier(img_input_dim)
        self.text_md_net = ModalClassifier(text_input_dim)
        self.num_classes = num_classes
        self.linear1 = nn.Linear(1024, 4096)
        self.linear2 = nn.Linear(1024,300)
        self.Relu = nn.ReLU()

        self.channles = in_channel
        self.aff_fusion = iAFF(1024, r=16)
        self.daf = DAF
        self.naff=NAFF(1024,r=16)
        if GNN == 'GAT':
            self.gnn = GraphAttentionLayer
        elif GNN == 'GCN':
            self.gnn = GraphConvolution
        else:
            raise NameError("Invalid GNN name!")
        self.n_layers = n_layers

        self.H = None
        self.H_text = None
        self.hyper_model = HGNN(in_ch=minus_one_dim, n_class=minus_one_dim, n_hid=minus_one_dim * 2, dropout=0.1)
        # transformer
        self.transformer = Transformer(minus_one_dim, 4, 2)  # 最后需要调参
        self.local_attention = Transformer(minus_one_dim, 2, 1)  # 当作注意力使用

        self.relu = nn.LeakyReLU(0.2)
        self.lrn = [self.gnn(in_channel, minus_one_dim)]  # 这可以尝试换成超图试试

        for i in range(1, self.n_layers):
            self.lrn.append(self.gnn(minus_one_dim, minus_one_dim))
        for i, layer in enumerate(self.lrn):
            self.add_module('lrn_{}'.format(i), layer)
        self.hypo = nn.Linear(self.n_layers * minus_one_dim, minus_one_dim)

        _adj = torch.FloatTensor(gen_A(num_classes, t, adj_file))
        if GNN == 'GAT':
            self.adj = Parameter(_adj, requires_grad=False)
        else:
            self.adj = Parameter(gen_adj(_adj), requires_grad=False)

        if inp is not None:
            self.inp = Parameter(inp, requires_grad=False)
        else:
            self.inp = Parameter(torch.rand(num_classes, in_channel))
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature_img, feature_text):
        view1_feature = self.img_net(feature_img)
        view2_feature = self.text_net(feature_text)
        # 全局
        global_img_feature = view1_feature
        # global_img_feature = global_img_feature.unsqueeze(0)
        # global_img_feature = self.transformer(global_img_feature)
        # global_img_feature = global_img_feature.squeeze(0)
        # global_img_feature = global_img_feature.cuda().data.cpu().detach().numpy()

        global_text_feature = view2_feature
        # global_text_feature = global_text_feature.unsqueeze(0)
        # global_text_feature = self.transformer(global_text_feature)
        # global_text_feature = global_text_feature.squeeze(0)
        # global_text_feature = global_text_feature.cuda().data.cpu().detach().numpy()

        # 局部
        local_img_feature = view1_feature
        local_text_feature = view2_feature

        local_img_feature = local_img_feature.unsqueeze(0)
        local_text_feature = local_text_feature.unsqueeze(0)

        local_img_feature = self.local_attention(local_img_feature)
        local_img_feature = local_img_feature.squeeze(0)
        local_text_feature = self.local_attention(local_text_feature)
        local_text_feature = local_text_feature.squeeze(0)

        local_img_feature = local_img_feature.unsqueeze(2)
        local_text_feature = local_text_feature.unsqueeze(2)
        local_img_feature = local_img_feature.unsqueeze(3)
        local_text_feature = local_text_feature.unsqueeze(3)

        fusion_fus_feature = self.aff_fusion(local_text_feature,local_img_feature)#local_text_feature+local_img_feature
        fusion_fus_feature = fusion_fus_feature.squeeze(3)
        fusion_fus_feature = fusion_fus_feature.squeeze(2)
        # 混合模块超图部分
        fusion_fus_feature = fusion_fus_feature.unsqueeze(0)
        fusion_fus_feature = self.transformer(fusion_fus_feature)
        fusion_fus_feature = fusion_fus_feature.squeeze(0)
        fusion_fus_feature = fusion_fus_feature.cuda().data.cpu().detach().numpy()
        tmp = hgut.construct_H_with_KNN(fusion_fus_feature, K_neigs=[1], split_diff_scale=False, is_probH=True,
                                        m_prob=1)
        new_H = hgut.hyperedge_concat(self.H, tmp)
        G = hgut.generate_G_from_H(new_H)
        G = torch.Tensor(G).cuda()
        fusion_fus_feature = torch.Tensor(fusion_fus_feature).cuda()
        fusion_fus_feature = self.hyper_model(fusion_fus_feature, G)

        local_img_feature = local_img_feature.squeeze(3)
        local_text_feature = local_text_feature.squeeze(3)
        local_img_feature = local_img_feature.squeeze(2)
        local_text_feature = local_text_feature.squeeze(2)

        # 图像模块超图部分
        # local_img_feature = torch.Tensor(local_img_feature)

        local_img_feature = local_img_feature.unsqueeze(0)
        local_img_feature = self.transformer(local_img_feature)
        local_img_feature = local_img_feature.squeeze(0)
        local_img_feature = local_img_feature.cuda().data.cpu().detach().numpy()
        tmp = hgut.construct_H_with_KNN(local_img_feature, K_neigs=[1], split_diff_scale=False, is_probH=True, m_prob=1)
        new_H = hgut.hyperedge_concat(self.H, tmp)
        G = hgut.generate_G_from_H(new_H)
        G = torch.Tensor(G).cuda()
        local_img_feature = torch.Tensor(local_img_feature).cuda()
        local_img_feature = self.hyper_model(local_img_feature, G)

        # 文本模块超图部分
        local_text_feature = local_text_feature.unsqueeze(0)
        local_text_feature = self.transformer(local_text_feature)
        local_text_feature = local_text_feature.squeeze(0)
        local_text_feature = local_text_feature.cuda().data.cpu().detach().numpy()
        tmp = hgut.construct_H_with_KNN(local_text_feature, K_neigs=[1], split_diff_scale=False, is_probH=True,
                                        m_prob=1)
        new_H = hgut.hyperedge_concat(self.H, tmp)
        G = hgut.generate_G_from_H(new_H)
        G = torch.Tensor(G).cuda()
        local_text_feature = torch.Tensor(local_text_feature).cuda()
        local_text_feature = self.hyper_model(local_text_feature, G)

        global_img_feature = global_img_feature.unsqueeze(2)
        local_img_feature = local_img_feature.unsqueeze(2)
        global_text_feature = global_text_feature.unsqueeze(2)
        local_text_feature = local_text_feature.unsqueeze(2)
        global_img_feature = global_img_feature.unsqueeze(3)
        local_img_feature = local_img_feature.unsqueeze(3)
        global_text_feature = global_text_feature.unsqueeze(3)
        local_text_feature = local_text_feature.unsqueeze(3)

        fusion_fus_feature = fusion_fus_feature.unsqueeze(2)
        fusion_fus_feature = fusion_fus_feature.unsqueeze(3)

        # fusion_img_feature = torch.cat((global_img_feature, local_img_feature), dim=1)
        # fusion_text_feature = torch.cat((global_text_feature, local_text_feature), dim=1)
        # fusion_img_feature=self.linear1(fusion_img_feature)
        # fusion_text_feature=self.linear1(fusion_text_feature)

        fusion_img_feature = self.aff_fusion(local_img_feature, global_img_feature)#local_img_feature+global_img_feature#global_img_feature
        fusion_text_feature = self.aff_fusion(local_text_feature, global_text_feature)#local_text_feature+global_text_feature#global_text_feature
        fusion_img_feature = self.aff_fusion(fusion_img_feature, fusion_fus_feature)##fusion_img_feature + fusion_fus_feature
        fusion_text_feature = self.aff_fusion(fusion_text_feature, fusion_fus_feature)##fusion_text_feature + fusion_fus_feature
        # fusion_img_feature = self.aff_fusion(local_img_feature, fusion_fus_feature)
        # fusion_text_feature = self.aff_fusion(local_text_feature, fusion_fus_feature)

        fusion_img_feature = fusion_img_feature.squeeze(3)
        fusion_img_feature = fusion_img_feature.squeeze(2)
        fusion_text_feature = fusion_text_feature.squeeze(3)
        fusion_text_feature = fusion_text_feature.squeeze(2)

        fusion_img_feature = fusion_img_feature.unsqueeze(0)
        fusion_img_feature = self.transformer(fusion_img_feature)
        view1_feature = fusion_img_feature.squeeze(0)
        # view1_feature = self.linear1(view1_feature)
        view1_feature = self.Relu(view1_feature)

        fusion_text_feature = fusion_text_feature.unsqueeze(0)
        fusion_text_feature = self.transformer(fusion_text_feature)
        view2_feature = fusion_text_feature.squeeze(0)
        # view2_feature = self.linear2(view2_feature)
        view2_feature = self.Relu(view2_feature)

        layers = []
        x = self.inp
        for i in range(self.n_layers):
            x = self.lrn[i](x, self.adj)
            if self.gnn == GraphConvolution:
                x = self.relu(x)
            layers.append(x)
        x = torch.cat(layers, -1)
        x = self.hypo(x)

        norm_img = torch.norm(view1_feature, dim=1)[:, None] * torch.norm(x, dim=1)[None, :] + 1e-6
        norm_txt = torch.norm(view2_feature, dim=1)[:, None] * torch.norm(x, dim=1)[None, :] + 1e-6
        x = x.transpose(0, 1)
        y_img = torch.matmul(view1_feature, x)
        y_text = torch.matmul(view2_feature, x)
        y_img = y_img / norm_img
        y_text = y_text / norm_txt

        view1_feature_view2 = self.img2text_net(view1_feature)
        view2_feature_view1 = self.text2img_net(view2_feature)
        # view1_feature = self.linear1(view1_feature)
        # view2_feature = self.linear2(view2_feature)
        view1_modal_view1 = self.img_md_net(feature_img)
        view2_modal_view1 = self.img_md_net(view2_feature_view1)
        view1_modal_view2 = self.text_md_net(view1_feature_view2)
        view2_modal_view2 = self.text_md_net(feature_text)

        return view1_feature, view2_feature, y_img, y_text, x.transpose(0, 1), \
               view1_modal_view1, view2_modal_view1, view1_modal_view2, view2_modal_view2

class HGcont(nn.Module):
    def __init__(self, img_input_dim=4096, text_input_dim=1024, minus_one_dim=1024, num_classes=10, in_channel=300,
                 t=0.4, k=3, inp=None, GNN='GAT', n_layers=4):
        super(HGcont, self).__init__()
        self.img_net = ImgNN(img_input_dim, minus_one_dim)
        self.text_net = TextNN(text_input_dim, minus_one_dim)
        self.img2text_net = TextDec(minus_one_dim, text_input_dim)
        self.text2img_net = ImgDec(minus_one_dim, img_input_dim)
        self.img_md_net = ModalClassifier(img_input_dim)
        self.text_md_net = ModalClassifier(text_input_dim)
        self.num_classes = num_classes
        self.linear1 = nn.Linear(2048, 1024)
        self.Relu = nn.ReLU()

        self.channles = in_channel
        self.aff_fusion = iAFF(1024,r=16)

        if GNN == 'GAT':
            self.gnn = GraphAttentionLayer
        elif GNN == 'GCN':
            self.gnn = GraphConvolution
        else:
            raise NameError("Invalid GNN name!")
        self.n_layers = n_layers
        # 超图模块构建
        self.H = None
        self.H_text = None
        self.hyper_model = HGNN(in_ch=minus_one_dim, n_class=minus_one_dim, n_hid=minus_one_dim * 2, dropout=0.1)

        # transformer
        self.transformer = Transformer(minus_one_dim , 4, 2)  # 最后需要调�?
        self.local_attention = Transformer(minus_one_dim, 2, 1)  # 当作注意力使�?
        self.relu = nn.LeakyReLU(0.2)
        self.lrn = [self.gnn(in_channel, minus_one_dim)]
        for i in range(1, self.n_layers):
            self.lrn.append(self.gnn(minus_one_dim, minus_one_dim))
        for i, layer in enumerate(self.lrn):
            # enumerate多用于在for循环中得到计数，利用它可以同时获得索引和值，即需要index和value值的时候可以使用enumerate
            self.add_module('lrn_{}'.format(i), layer)
        self.hypo = nn.Linear(self.n_layers * minus_one_dim, minus_one_dim)

        if inp is None:
            raise NotImplementedError("Category embeddings are missing!")
        self.inp = Parameter(inp, requires_grad=True)
        # Parameter函数可以对某个张量进行参数化。它可以将不可训练的张量转化为可训练的参数类型，
        # 同时将转化后的张量绑定到模型可训练参数的列表中，当更新模型的参数时一并将其更新�?
        normalized_inp = F.normalize(inp, dim=1)
        # F.normalize(data, p=2 / 1, dim=0 / 1 / -1)
        # 将某一个维度除以那个维度对应的范数(默认�?范数)
        # data: 输入的数据（tensor）p: L2 / L1_norm运算
        # dim: 0表示按列操作，则每列都是除以该列下平方和的开方；1表示按行操作，则每行都是除以该行下所有元素平方和的开�?
        self.A0 = Parameter(torch.matmul(normalized_inp, normalized_inp.T), requires_grad=False)
        self.A0[self.A0 < t] = 0  #
        self.t = t

        self.Wk = list()
        self.k = k
        for i in range(k):
            Wk_temp = Parameter(torch.zeros(in_channel))
            torch.nn.init.uniform_(Wk_temp.data, 0.3, 0.6)
            # torch.nn.init.uniform_(tensor, a=0.0, b=1.0)从均匀分布U(a, b)中生成值，填充输入的张量或变量�?
            # 参数tensor n维的torch.Tensor a 均匀分布的下�?b 均匀分布的上�?
            self.Wk.append(Wk_temp)
        for i, Wk_temp in enumerate(self.Wk):
            self.register_parameter('W_{}'.format(i), Wk_temp)

        self.lambdda = 0.5

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature_img, feature_text):
        view1_feature = self.img_net(feature_img)
        view2_feature = self.text_net(feature_text)

        # 全局
        global_img_feature = view1_feature
        global_text_feature = view2_feature

        # 局部
        local_img_feature = view1_feature
        local_text_feature = view2_feature

        local_img_feature = local_img_feature.unsqueeze(0)
        local_text_feature = local_text_feature.unsqueeze(0)

        local_img_feature = self.local_attention(local_img_feature)
        local_img_feature = local_img_feature.squeeze(0)
        local_text_feature = self.local_attention(local_text_feature)
        local_text_feature =local_text_feature.squeeze(0)

        local_img_feature = local_img_feature.unsqueeze(2)
        local_text_feature = local_text_feature.unsqueeze(2)
        local_img_feature = local_img_feature.unsqueeze(3)
        local_text_feature = local_text_feature.unsqueeze(3)
        fusion_fus_feature = self.aff_fusion(local_img_feature, local_text_feature)
        fusion_fus_feature = fusion_fus_feature.squeeze(3)
        fusion_fus_feature = fusion_fus_feature.squeeze(2)
        # 混合模块超图部分
        fusion_fus_feature = fusion_fus_feature.unsqueeze(0)
        fusion_fus_feature = self.transformer(fusion_fus_feature)
        fusion_fus_feature = fusion_fus_feature.squeeze(0)
        fusion_fus_feature = fusion_fus_feature.cuda().data.cpu().detach().numpy()
        tmp = hgut.construct_H_with_KNN(fusion_fus_feature, K_neigs=[3], split_diff_scale=False, is_probH=True, m_prob=1)
        new_H = hgut.hyperedge_concat(self.H, tmp)
        G = hgut.generate_G_from_H(new_H)
        G = torch.Tensor(G).cuda()
        fusion_fus_feature = torch.Tensor(fusion_fus_feature).cuda()
        fusion_fus_feature = self.hyper_model(fusion_fus_feature, G)

        local_img_feature = local_img_feature.squeeze(3)
        local_text_feature = local_text_feature.squeeze(3)
        local_img_feature = local_img_feature.squeeze(2)
        local_text_feature = local_text_feature.squeeze(2)

        # 图像模块超图部分
        # local_img_feature = torch.Tensor(local_img_feature)
        local_img_feature = local_img_feature.unsqueeze(0)
        local_img_feature = self.transformer(local_img_feature)
        local_img_feature = local_img_feature.squeeze(0)
        local_img_feature = local_img_feature.cuda().data.cpu().detach().numpy()
        tmp = hgut.construct_H_with_KNN(local_img_feature, K_neigs=[4],split_diff_scale=False,is_probH=True, m_prob=1)
        new_H = hgut.hyperedge_concat(self.H, tmp)
        G = hgut.generate_G_from_H(new_H)
        G = torch.Tensor(G).cuda()
        local_img_feature = torch.Tensor(local_img_feature).cuda()
        local_img_feature = self.hyper_model(local_img_feature, G)

        # 文本模块超图部分
        local_text_feature = local_text_feature.unsqueeze(0)
        local_text_feature = self.transformer(local_text_feature)
        local_text_feature = local_text_feature.squeeze(0)
        local_text_feature = local_text_feature.cuda().data.cpu().detach().numpy()
        tmp = hgut.construct_H_with_KNN(local_text_feature, K_neigs=[2], split_diff_scale=False, is_probH=True, m_prob=1)
        new_H = hgut.hyperedge_concat(self.H, tmp)
        G = hgut.generate_G_from_H(new_H)
        G = torch.Tensor(G).cuda()
        local_text_feature = torch.Tensor(local_text_feature).cuda()
        local_text_feature = self.hyper_model(local_text_feature, G)

        # fusion_img_feature = torch.cat((global_img_feature, local_img_feature), dim=-1)
        # fusion_text_feature = torch.cat((global_text_feature, local_text_feature), dim=-1)
        global_img_feature = global_img_feature.unsqueeze(2)
        local_img_feature = local_img_feature.unsqueeze(2)
        global_text_feature = global_text_feature.unsqueeze(2)
        local_text_feature = local_text_feature.unsqueeze(2)
        global_img_feature = global_img_feature.unsqueeze(3)
        local_img_feature = local_img_feature.unsqueeze(3)
        global_text_feature = global_text_feature.unsqueeze(3)
        local_text_feature = local_text_feature.unsqueeze(3)

        fusion_fus_feature = fusion_fus_feature.unsqueeze(2)
        fusion_fus_feature = fusion_fus_feature.unsqueeze(3)

        fusion_img_feature = self.aff_fusion(global_img_feature, local_img_feature)
        fusion_text_feature = self.aff_fusion(global_text_feature, local_text_feature)
        fusion_img_feature = self.aff_fusion(fusion_img_feature, fusion_fus_feature)
        fusion_text_feature = self.aff_fusion(fusion_text_feature, fusion_fus_feature)

        fusion_img_feature = fusion_img_feature.squeeze(3)
        fusion_img_feature = fusion_img_feature.squeeze(2)
        fusion_text_feature = fusion_text_feature.squeeze(3)
        fusion_text_feature = fusion_text_feature.squeeze(2)

        fusion_img_feature = fusion_img_feature.unsqueeze(0)
        fusion_img_feature = self.transformer(fusion_img_feature)
        view1_feature = fusion_img_feature.squeeze(0)
        # view1_feature = self.linear1(view1_feature)
        view1_feature = self.Relu(view1_feature)

        fusion_text_feature = fusion_text_feature.unsqueeze(0)
        fusion_text_feature = self.transformer(fusion_text_feature)
        view2_feature = fusion_text_feature.squeeze(0)
        # view2_feature = self.linear1(view2_feature)
        view2_feature = self.Relu(view2_feature)

        S = torch.zeros_like(self.A0)
        for Wk_temp in self.Wk:
            normalized_imp_mul_Wk = F.normalize(self.inp * Wk_temp[None, :], dim=1)
            S += torch.matmul(normalized_imp_mul_Wk, normalized_imp_mul_Wk.T)
        S /= self.k
        S[S < self.t] = 0
        A = self.lambdda * self.A0 + (1 - self.lambdda) * S
        adj = gen_adj(A)

        layers = []
        x = self.inp
        for i in range(self.n_layers):
            x = self.lrn[i](x, adj)
            if self.gnn == GraphConvolution:
                x = self.relu(x)
            layers.append(x)
        x = torch.cat(layers, -1)  # -1按列相加
        x = self.hypo(x)

        norm_img = torch.norm(view1_feature, dim=1)[:, None] * torch.norm(x, dim=1)[None, :] + 1e-6
        norm_txt = torch.norm(view2_feature, dim=1)[:, None] * torch.norm(x, dim=1)[None, :] + 1e-6
        # torch.norm返回输入张量给定维dim 上每行的p 范数�?
        x = x.transpose(0, 1)
        y_img = torch.matmul(view1_feature, x)
        y_text = torch.matmul(view2_feature, x)
        y_img = y_img / norm_img
        y_text = y_text / norm_txt

        view1_feature_view2 = self.img2text_net(view1_feature)
        view2_feature_view1 = self.text2img_net(view2_feature)
        view1_modal_view1 = self.img_md_net(feature_img)
        view2_modal_view1 = self.img_md_net(view2_feature_view1)
        view1_modal_view2 = self.text_md_net(view1_feature_view2)
        view2_modal_view2 = self.text_md_net(feature_text)

        return view1_feature, view2_feature, y_img, y_text, x.transpose(0, 1), \
               view1_modal_view1, view2_modal_view1, view1_modal_view2, view2_modal_view2
