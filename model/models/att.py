import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

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

def store_class_memory(depth_feature_maps, label_map, num_classes):

    #只取前5个类别
    #第一种：只取最后的feature
    #第二种：取多尺度feature
    multi_scale_class_memory = []
    with torch.no_grad():
        # for j, (depth_feature_map, class_memory_bank) in enumerate(zip(depth_feature_maps, multi_scale_class_memory)):
            # depth_feature_map = depth_feature_maps[0]
        for depth_feature_map in depth_feature_maps:
            class_memory_bank = torch.zeros((depth_feature_map.size(0), num_classes, 128)).cuda()
            batch_size, num_channels, height, width = depth_feature_map.shape
            _, height, width = label_map.shape
            embed_dims = num_channels
            # 初始化一个数组用于存储每个类别的Prototype
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
                    if mask.sum() == 0:
                        prototype = torch.zeros((embed_dims)).to(depth_feature_map.device)
                    else:
                        prototype = torch.sum(depth_feature * mask, dim=(1, 2)) / torch.sum(mask, dim=(1, 2))
                    # prototype = prototype.unsqueeze(-1).unsqueeze(-1)
                    # 将Prototype添加到类别内存库中
                    class_memory_bank[i, class_idx] += prototype
                    # with torch.no_grad():
                    #
                    #     class_memory_bank[i, class_idx] = 0.95 * class_memory_bank[i, class_idx] + (1 - 0.95) * prototype
            # multi_scale_class_memory[j] = class_memory_bank

        # class_memory_bank[i, class_idx] += prototype
            multi_scale_class_memory.append(class_memory_bank.sum(axis=0).unsqueeze(0) / batch_size)

    return multi_scale_class_memory

def store_class_memory_ema(depth_feature_maps, label_map, num_classes, multi_scale_class_memory, idx):

    #只取前5个类别
    #第一种：只取最后的feature
    #第二种：取多尺度feature
    # with torch.no_grad():
    for j, (depth_feature_map, class_memory_bank) in enumerate(zip(depth_feature_maps, multi_scale_class_memory)):
            # depth_feature_map = depth_feature_maps[0]
        # for depth_feature_map in depth_feature_maps:
        #     class_memory_bank = torch.zeros((depth_feature_map.size(0), num_classes, 128)).cuda()
            batch_size, num_channels, height, width = depth_feature_map.shape
            _, height, width = label_map.shape
            embed_dims = num_channels
            # 初始化一个数组用于存储每个类别的Prototype
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
                    if mask.sum() == 0:
                        prototype = torch.zeros((embed_dims)).to(depth_feature_map.device)
                    else:
                        prototype = torch.sum(depth_feature * mask, dim=(1, 2)) / torch.sum(mask, dim=(1, 2))
                    # prototype = prototype.unsqueeze(-1).unsqueeze(-1)
                    # 将Prototype添加到类别内存库中
                    # class_memory_bank[i, class_idx] += prototype
                    # with torch.no_grad():
                    if idx == 0:
                        with torch.no_grad():
                            class_memory_bank[0, class_idx] += prototype
                    else:
                        with torch.no_grad():
                            class_memory_bank[0, class_idx] = 0.95 * class_memory_bank[0, class_idx] + (1 - 0.95) * prototype
                # multi_scale_class_memory[j] = class_memory_bank

            # class_memory_bank[i, class_idx] += prototype
            #     multi_scale_class_memory.append(class_memory_bank.sum(axis=0).unsqueeze(0) / batch_size)
    if idx==0:
        return multi_scale_class_memory / batch_size

    return multi_scale_class_memory


def get_adaptive_threshold(depth_feature_maps, label_map, num_classes):
    pass