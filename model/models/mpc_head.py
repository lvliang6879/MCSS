# Copyright (c) OpenMMLab. All rights reserved.
from tkinter.tix import Tree
import numpy as np
import torch
import torch.nn as nn

from mmcv.cnn import ConvModule
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.runner import ModuleList
from mmcv.cnn.utils.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmseg.models.backbones.vit import TransformerEncoderLayer
from mmseg.models.decode_heads.decode_head_memory import BaseDecodeHead_momory
from mmseg.ops import Upsample, resize
from mmseg.models.builder import HEADS
# from model.SEUNet import *
import torch.nn as nn
from torch import optim
# from eval import evluation, SemiSupervised_visualization, MT_evluation
from segmentation.utils import rgb_to_labels, load_files, get_imgs_labels, split_data
from torch.utils.tensorboard import SummaryWriter
from segmentation.semi_dataloader import *
from segmentation.model.SEUNet import se_resnext50_32x4d
from PIL import Image
import os
import torch

# from .decode_head import BaseDecodeHead
import matplotlib.pyplot as plt

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
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)

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



@HEADS.register_module()
class MPC_Head(BaseDecodeHead_momory):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, **kwargs):
        super(MPC_Head, self).__init__(
            input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))

        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        num_heads = 2
        embed_dims = 128
        # self.dec_proj = nn.Linear(in_channels, embed_dims)\

        self.num_subclasses = 4
        self.cls_emb = nn.Parameter(
            torch.randn(1, self.num_classes*self.num_subclasses, embed_dims))
        for i in range(len(feature_strides)-1):
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=embed_dims,
                    nhead=num_heads,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                        d_model=embed_dims,
                        nhead=num_heads,
                        dropout=0.0,
                        normalize_before=False,
                    )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=embed_dims,
                    dim_feedforward=num_heads,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

        norm_cfg = dict(type='LN')


        self.patch_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.classes_proj = nn.Linear(embed_dims, embed_dims, bias=False)

        self.mask_norm = build_norm_layer(
            norm_cfg, self.num_classes*self.num_subclasses, postfix=2)[1]

        delattr(self, 'conv_seg')
        init_std = 0.02
        self.init_std = init_std
 
    def init_weights(self):
        trunc_normal_(self.cls_emb, std=self.init_std)
        trunc_normal_init(self.patch_proj, std=self.init_std)
        trunc_normal_init(self.classes_proj, std=self.init_std)
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=self.init_std, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def class_memory_bank(self, depth_feature_map, label_map, num_classes):
        batch_size, num_channels, height, width = depth_feature_map.shape
        _, height, width = label_map.shape
        embed_dims = num_channels

        # 初始化一个数组用于存储每个类别的Prototype
        prototypes = torch.zeros((batch_size, num_classes, embed_dims, 1, 1)).to(depth_feature_map.device)

        for i in range(batch_size):
            # 获取当前图像的深度特征和对应的标签图
            depth_feature = depth_feature_map[i]
            label = label_map[i]
            depth_feature = F.interpolate(depth_feature.unsqueeze(0), size=(height, width), mode='bilinear',
                                          align_corners=True).squeeze(0)

            # 遍历每个类别
            for class_idx in range(num_classes):
                # 创建当前类别的掩码
                mask = (label == class_idx).unsqueeze(0).expand_as(depth_feature)

                # 对深度特征进行masked average pooling，计算当前类别的Prototype
                prototype = torch.sum(depth_feature * mask, dim=(1, 2)) / torch.sum(mask, dim=(1, 2))
                prototype = prototype.unsqueeze(-1).unsqueeze(-1)

                # 将Prototype添加到类别内存库中
                prototypes[i, class_idx] += prototype

        # 计算类别内存库中Prototype的平均值，得到最终的类别内存库

        return prototypes.sum(dim=0) / batch_size




    def forward(self, deep_features):

        x = deep_features

        output = self.scale_heads[0](x[0])
        # output = x[0]
        # multi_prototype = [self.cls_emb.expand(output.size(0), -1, -1)]*len(self.feature_strides)
        class_memory_bank = self.class_memory_bank()
        resized_patches = output
        b, c, h, w = resized_patches.shape
        resized_patches = resized_patches.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
        aug_features = self.transformer_cross_attention_layers(resized_patches, class_memory_bank)
        aug_features = self.transformer_ffn_layers(aug_features)


        for i in range(1, len(self.feature_strides)):
            # non inplace
            resized_patches = output
            b, c, h, w = resized_patches.shape
            resized_patches = resized_patches.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
            multi_prototype[i] = self.transformer_cross_attention_layers[i-1](multi_prototype[i-1], resized_patches)
            multi_prototype[i] = self.transformer_self_attention_layers[i-1](multi_prototype[i])
            multi_prototype[i] = self.transformer_ffn_layers[i-1](multi_prototype[i])

            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        b, c, h, w = output.shape
        output = output.permute(0, 2, 3, 1).contiguous().view(b, -1, c)

        output = self.patch_proj(output)
        cls_seg_feat = self.classes_proj(multi_prototype[-1])

        output = F.normalize(output, dim=2, p=2)
        cls_seg_feat = F.normalize(cls_seg_feat, dim=2, p=2)

        output = output @ cls_seg_feat.transpose(1, 2)

        output = self.mask_norm(output)
        output = output.permute(0, 2, 1).contiguous().view(b,self.num_subclasses,-1, h, w)

        return output, multi_prototype




if __name__ == '__main__':
    import sys
    sys.path.append('/home/lvliang/project/mmsegmentation-mpcnet-main/mmsegmentation-mpcnet-main/mmseg')

    test_root = '/home/lvliang/project/dataset/Vaihingen/crop/image/'
    test_files = os.listdir(test_root)
    test_files.sort()
    testLabel_root = '/home/lvliang/project/dataset/Vaihingen/crop/class_label/'
    test_labels = os.listdir(testLabel_root)
    test_labels.sort()
    all_kappa = []
    all_f1 = []
    all_accu = []
    all_iou = []

    data_files = load_files(test_root)
    label_files = load_files(testLabel_root)
    # train_files, train_label_files, val_files, val_label_files = split_data(data_files, label_files, 0, 0)
    imgs, labels = get_imgs_labels(test_root, testLabel_root, data_files, label_files)

    net = se_resnext50_32x4d(num_classes=6, pretrained=None)
    net.cuda()
    # net.load_state_dict(torch.load(
    #     '/data1/users/lvliang/project/UniMatch-main/exp/isprs/supervised/r50/full_supervised/SEUNet_1_4.pth'))
    # 设置dataloader
    test_dataset = TestDataset(imgs, labels)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True, drop_last=True, num_workers=0)

    # former = SegmentationModel(num_classes=6, d_model=64)
    former = MPC_Head(
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    former.cuda()
    inpit_features = []
    for x_batch, y_batch in test_loader:
        imgs = x_batch.float().cuda()
        y_batch = y_batch.cuda()
        # masks_pred, deep_features = net(imgs)
        # num_classes = 5
        # prototypes = construct_prototype(deep_features, y_batch, num_classes)
        # former(deep_features)
        input_image = torch.randn(2, 8, 256, 256).cuda() # Batch size: 8, Input image size: 256x256
        inpit_features.append(torch.randn(2, 256, 256, 256).cuda())
        inpit_features.append(torch.randn(2, 256, 256, 256).cuda())
        inpit_features.append(torch.randn(2, 256, 256, 256).cuda())
        inpit_features.append(torch.randn(2, 256, 256, 256).cuda())
        output = former(inpit_features)
