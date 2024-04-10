import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
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
    # multi_scale_class_memory = torch.zeros((4, 1, 6, 128)).cuda()
    for j, depth_feature_map in  enumerate(depth_feature_maps):
        # depth_feature_map = depth_feature_maps[0]
        # for depth_feature_map in depth_feature_maps:
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
        multi_scale_class_memory.append(class_memory_bank.sum(axis=0).unsqueeze(0) / batch_size)
        stacked_tensor = torch.stack(multi_scale_class_memory, dim=0)

    return stacked_tensor.cuda()

def store_class_memory_ema(args, depth_feature_maps, label_map, num_classes, multi_scale_class_memory, idx):

    #只取前5个类别
    #第一种：只取最后的feature
    #第二种：取多尺度feature
    for j, (depth_feature_map, class_memory_bank) in enumerate(zip(depth_feature_maps, multi_scale_class_memory)):
            # depth_feature_map = depth_feature_maps[0]
        # for depth_feature_map in depth_feature_maps:
        #     class_memory_bank1 = torch.zeros((depth_feature_map.size(0), num_classes, 128)).cuda()
            batch_size, num_channels, height, width = depth_feature_map.shape
            _, height, width = label_map.shape
            embed_dims = num_channels
            # 初始化一个数组用于存储每个类别的Prototype
            for i in range(batch_size):
                # 获取当前图像的深度特征和对应的标签图
                depth_feature = depth_feature_map[i]
                label = label_map[i]
                depth_feature = F.interpolate(depth_feature.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=True).squeeze(0)
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
                    if idx == 0:
                        with torch.no_grad():
                            class_memory_bank[0, class_idx] += prototype
                    else:
                        with torch.no_grad():
                            class_memory_bank[0, class_idx] = args.SGM_lamda * class_memory_bank[0, class_idx] + (1 - args.SGM_lamda) * prototype
            # multi_scale_class_memory[j] = class_memory_bank
    if idx == 0:
        multi_scale_class_memory = multi_scale_class_memory / batch_size
        # cluster_center = online_cluster(multi_scale_class_memory, 128)
        # return multi_scale_class_memory / batch_size
        return multi_scale_class_memory

        # class_memory_bank[i, class_idx] += prototype
        #     multi_scale_class_memory.append(class_memory_bank.sum(axis=0).unsqueeze(0) / batch_size)

    return multi_scale_class_memory



def online_cluster(class_prototype_memory, feature_dim):
    # 将原型数据扁平化，以便进行 K-Means 聚类
    flattened_prototypes = class_prototype_memory.reshape(-1, feature_dim)

    # 设置 K-Means 聚类的簇数（每个类别内的子簇数量）
    num_clusters = 4

    prototypes = flattened_prototypes.cpu().numpy()

    # 使用 K-Means 进行聚类
    kmeans = KMeans(n_clusters=48, random_state=0)
    kmeans.fit(prototypes)

    # 获取聚类后的中心（新的原型）
    cluster_centers = kmeans.cluster_centers_

    # 将聚类后的中心重新组织为类别原型内存的形状
    new_class_prototype_memory = torch.tensor(cluster_centers.reshape( num_clusters, 12,feature_dim)).cuda().unsqueeze(1)
    # cluster_centers = torch.tensor(cluster_centers).cuda().unsqueeze(0)

    return new_class_prototype_memory





