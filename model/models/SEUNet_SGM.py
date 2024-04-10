"""
ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
from __future__ import print_function, division, absolute_import
from collections import OrderedDict
import math

import torch.nn as nn
from torch.utils import model_zoo
import torch
import torch.nn.functional as F
from segmentation.model.basic import _ConvBNReLU
# from att import *

__all__ = ['SENet', 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152',
           'se_resnext50_32x4d', 'se_resnext101_32x4d']

pretrained_settings = {
    'senet154': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet50': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet101': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet152': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext50_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext101_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
}

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask = None,
                     tgt_key_padding_mask = None,
                     query_pos= None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask = None,
                    tgt_key_padding_mask = None,
                    query_pos = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask = None,
                tgt_key_padding_mask= None,
                query_pos = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout ,batch_first=True)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask = None,
                     memory_key_padding_mask = None,
                     pos  = None,
                     query_pos = None):


        tgt2 = self.multihead_attn(query=tgt, key=memory, value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask = None,
                    memory_key_padding_mask= None,
                    pos= None,
                    query_pos = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory):
        if self.normalize_before:
            return self.forward_pre(tgt, memory)
        return self.forward_post(tgt, memory)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = nn.ReLU()
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)



class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000, **kwargs):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        # self.avg_pool = nn.AvgPool2d(7, stride=1)
        # self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        # self.last_linear = nn.Linear(512 * block.expansion, num_classes)

        self.up2_conv_reduce = _ConvBNReLU(512, 64, 1, **kwargs)
        self.up3_conv_reduce = _ConvBNReLU(1024, 256, 1, **kwargs)
        self.up4_conv_reduce = _ConvBNReLU(2048, 512, 1, **kwargs)
        self.up5_conv_reduce = _ConvBNReLU(2048, 1024, 1, **kwargs)

        # self.relu = nn.ReLU(inplace=True)
        # self.deconv1 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.bn1 = nn.BatchNorm2d(32)
        # self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.deconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.bn4 = nn.BatchNorm2d(512)
        # self.deconv5 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        # self.bn5 = nn.BatchNorm2d(1024)
        # self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)
        self.num_classes = num_classes

        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)



        self.up2p_conv_reduce = _ConvBNReLU(256, 128, 1, **kwargs)
        self.up3p_conv_reduce = _ConvBNReLU(512, 128, 1, **kwargs)
        self.up4p_conv_reduce = _ConvBNReLU(1024, 128, 1, **kwargs)
        self.up5p_conv_reduce = _ConvBNReLU(2048, 128, 1, **kwargs)
        self.seg_conv_out = nn.Conv2d(128, num_classes, 1)

        num_heads = 2
        embed_dims = 128
        self.classes_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.bn_classes_proj = nn.BatchNorm1d(5)

        self.cls_emb = nn.Parameter(
            torch.randn(1, num_classes, embed_dims))


        self.transformer_self_attention_layers =  SelfAttentionLayer(
                d_model=embed_dims,
                nhead=num_heads,
                dropout=0.0,
                normalize_before=False,
            )
        self.transformer_cross_attention_layers = CrossAttentionLayer(
                d_model=embed_dims,
                nhead=num_heads,
                dropout=0.0,
                normalize_before=False,
            )
        self.transformer_ffn_layers =  FFNLayer(
                d_model=embed_dims,
                dim_feedforward=num_heads,
                dropout=0.0,
                normalize_before=False,)



        in_channels = 256
        # self.cls_emb = nn.Parameter(
        #     torch.randn(1, self.num_classes * self.num_subclasses, embed_dims))
        # self.transformer_cross_attention_layers.append(
        #     CrossAttentionLayer(
        #         d_model=embed_dims,
        #         nhead=num_heads,
        #         dropout=0.0,
        #         normalize_before=False,
        #     )
        # )
        # self.transformer_self_attention_layers.append(
        #     SelfAttentionLayer(
        #         d_model=embed_dims,
        #         nhead=num_heads,
        #         dropout=0.0,
        #         normalize_before=False,
        #     )
        # )
        # self.transformer_ffn_layers.append(
        #     FFNLayer(
        #         d_model=embed_dims,
        #         dim_feedforward=num_heads,
        #         dropout=0.0,
        #         normalize_before=False,
        #     )
        # )

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)


        return x1, x2, x3, x4, x5

    def decoder(self, x1, x2, x3, x4, x5):
        """
        :param x2: c = 256
        :param x3: c = 512
        :param x4: c = 1024
        :param x5: c = 2048
        :return:
        """

        up = F.interpolate(self.up5_conv_reduce(x5), scale_factor=2, mode='bilinear')
        up = F.interpolate(self.up4_conv_reduce(torch.cat([up, x4], dim=1)), scale_factor=2, mode='bilinear', align_corners=True)
        up = F.interpolate(self.up3_conv_reduce(torch.cat([up, x3], dim=1)), scale_factor=2, mode='bilinear', align_corners=True)
        up = F.interpolate(self.up2_conv_reduce(torch.cat([up, x2], dim=1)), scale_factor=2, mode='bilinear', align_corners=True)
        seg_out = self.seg_conv_out(torch.cat([up, F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)], dim=1))
        # edge_out = self.edge_conv_out(torch.cat([up, F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)], dim=1))
        seg_out = F.interpolate(seg_out, scale_factor=2, mode='bilinear', align_corners=True)
        # edge_out = F.interpolate(edge_out, scale_factor=2, mode='bilinear', align_corners=True)
        return seg_out

    def decoder_ConvTranspose(self, x1, x2, x3, x4, x5, refined_features=None):
        """
        :param x2: c = 256
        :param x3: c = 512
        :param x4: c = 1024
        :param x5: c = 2048
        :return:
        """


        score = self.bn5(self.relu(self.deconv5(x5)))
        score = score + x4
        score = self.bn4(self.relu(self.deconv4(score)))
        score = score + x3
        score = self.bn3(self.relu(self.deconv3(score)))
        score = score + x2
        score = self.bn2(self.relu(self.deconv2(score)))
        # if refined_features != None:
        #     score = torch.cat((score, F.interpolate(refined_features[-1],score.shape[-2:], mode='bilinear')),dim=1)
        score = self.bn1(self.relu(self.deconv1(score)))

            # score += refined_features
        if refined_features != None:
            seg_out = self.classifier(score)
            for refined_feature in refined_features:
                seg_out += F.interpolate(refined_feature, (512, 512), mode='bilinear')
        else:
            seg_out = self.classifier(score)

        return seg_out


    def CRT(self, features, class_memory):
        refined_features =[]
        b, _, _,_ = features.shape

        # class_memory = self.bn_classes_proj(self.relu(self.classes_proj(class_memory)))
        # class_memory = self.cls_emb

        class_memory = class_memory.repeat(b, 1, 1)
        # for feature in features:
        feature = features
        b, c, h, w = feature.shape
        resized_patches = feature.permute(0, 2, 3, 1)
        # refined_feature = torch.cosine_similarity(resized_patches, class_memory)
        refined_feature = torch.zeros(b, 6, h, w).cuda()
        for i in range(b):
            for j in range(6):
                refined_feature[i, j] = torch.cosine_similarity(resized_patches[i], class_memory[i:i+1, j:j+1], dim=-1)
        # refined_feature = self.transformer_cross_attention_layers(resized_patches, class_memory)
        # refined_feature = self.transformer_self_attention_layers(refined_feature)
        # refined_feature = self.transformer_ffn_layers(refined_feature)

        # refined_feature = refined_feature.permute(0, 2, 1)

        # Reshape back to spatial dimensions
        # refined_feature = refined_feature.view(b, c, h, w)

        # refined_features.append(refined_feature.cuda())

        # 计算类别内存库中Prototype的平均值，得到最终的类别内存库
        return refined_feature


    def refine_features(self, multi_scale_features, multi_scale_class_memory):
        refined_features =[]
        for feature, class_memory in zip(multi_scale_features, multi_scale_class_memory):
            b, _, _,_ = feature.shape
            class_memory = class_memory.repeat(b, 1, 1)
            # feature = features[0]
            b, c, h, w = feature.shape
            resized_patches = feature.permute(0, 2, 3, 1).view(b, -1, c)
            # refined_feature = torch.cosine_similarity(resized_patches, class_memory)
            refined_feature = torch.cosine_similarity(resized_patches.unsqueeze(2), class_memory.unsqueeze(1), dim=-1)
            refined_feature = refined_feature.permute(0, 2, 1)
            refined_feature = refined_feature.view(b, self.num_classes, h, w)
            refined_features.append(refined_feature)
        return refined_features


    def forward(self, x, class_features = None):
        x1, x2, x3, x4, x5 = self.features(x)
        features = []
        # x5p = F.interpolate(self.up5p_conv_reduce(x5), x.shape[-2:], mode='bilinear')
        x5 = self.up5p_conv_reduce(x5)
        features.append(x5)
        # x4p = F.interpolate(self.up4p_conv_reduce(x4), x.shape[-2:], mode='bilinear')
        x4 = self.up4p_conv_reduce(x4)
        features.append(x4)
        # x3p = F.interpolate(self.up3p_conv_reduce(x3), x.shape[-2:], mode='bilinear')
        x3 = self.up3p_conv_reduce(x3)
        features.append(x3)
        # x2p = F.interpolate(self.up2p_conv_reduce(x2), x.shape[-2:], mode='bilinear')
        x2 = self.up2p_conv_reduce(x2)
        features.append(x2)
        # seg_out = self.decoder(x1las, x2, x3, x4, x5)


        if class_features != None:
            # refined_features = self.CRT(features, class_features)#CRT class refined transformer
            refined_features = self.refine_features(features, class_features)
            seg_out = self.decoder_ConvTranspose(x1, x2, x3, x4, x5, refined_features)
            # seg_out = self.decoder_ConvTranspose(x1, x2p, x3p, x4p, x5p)
            return seg_out
        # seg_out = self.decoder_ConvTranspose(x1, x2, x3, x4, x5)
        seg_out = self.decoder_ConvTranspose(x1, x2, x3, x4, x5)
        return seg_out, features




def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'], \
        'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']


def senet154(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16,
                  dropout_p=0.2, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['senet154'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet50(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet50'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet101(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 4, 23, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet101'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet152(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 8, 36, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet152'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet( SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext50_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnext101_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext101_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model

if __name__ == '__main__':
    net = se_resnext50_32x4d(6, pretrained=None)

    input = torch.rand([5, 3, 512, 512])
    input.float().cuda()
    out = net(input)