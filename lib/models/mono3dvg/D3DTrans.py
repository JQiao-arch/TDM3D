from typing import Optional, List
import math
import copy

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from utils.misc import inverse_sigmoid
from .ops.modules import MSDeformAttn


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class D3DTransformer(nn.Module):
    def __init__(
            self,
            d_model=256,
            nhead=8,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            return_intermediate_dec=False,
            num_feature_levels=4,
            dec_n_points=4,
            text_guided_adapter=None):

        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        decoder_layer = DepthAwareDecoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, dec_n_points)
        self.decoder = DepthAwareDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)


        self._reset_parameters()

        self.text_guided_adapter = text_guided_adapter

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self,
                intermediate_output,
                query_embeds,
                depth_pos_embed,
                mask_depth,
                text_memory=None,
                text_mask=None):

        memory = intermediate_output['memory']
        spatial_shapes = intermediate_output['spatial_shapes']
        level_start_index = intermediate_output['level_start_index']
        valid_ratios = intermediate_output['valid_ratios']
        mask_flatten = intermediate_output['mask_flatten']
        reference_points = intermediate_output['reference_points']
        pos_embed_flatten = intermediate_output['pos_embed_flatten']

        if self.text_guided_adapter is not None:
            _, depth_feat_for_3d = self.text_guided_adapter(
                memory,
                mask_flatten,
                pos_embed_flatten,
                reference_points,
                spatial_shapes,
                level_start_index,
                valid_ratios,
                text_memory,
                text_mask,
                depth_pos_embed,
                mask_depth,
                word_pos=None
            )
            _, depth_adapt_k = depth_feat_for_3d.chunk(2, dim=-1)
        else:
            depth_adapt_k = depth_pos_embed

        query_pos = query_embeds
        tgt = torch.zeros_like(query_pos)
        init_reference_out = reference_points

        hs, inter_references = self.decoder(
            tgt,
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            query_pos,
            mask_flatten,
            depth_adapt_k,
            mask_depth,
            text_memory,
            text_mask
        )

        inter_references_out = inter_references
        if self.decoder.return_intermediate:
            reference_points_out = inter_references[-1]
        else:
            reference_points_out = inter_references

        output = {
            'hs': hs,
            'init_reference_out': init_reference_out,
            'inter_references_out': inter_references_out,
            'reference_points': reference_points_out,
            'memory': memory,
            'spatial_shapes': spatial_shapes,
            'level_start_index': level_start_index,
            'valid_ratios': valid_ratios,
            'mask_flatten': mask_flatten
        }

        return output


class DepthAwareDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # depth cross attention
        self.cross_attn_depth = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_depth = nn.Dropout(dropout)
        self.norm_depth = nn.LayerNorm(d_model)

        # text cross attention
        self.cross_attn_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_text = nn.Dropout(dropout)
        self.norm_text = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)


    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self,
                tgt,
                query_pos,
                reference_points,
                src,
                src_spatial_shapes,
                level_start_index,
                src_padding_mask,
                depth_pos_embed,
                mask_depth,
                text_memory=None,
                text_mask=None):

        if depth_pos_embed is not None:
            if depth_pos_embed.dim() == 4:
                depth_memory = depth_pos_embed.flatten(2).permute(2, 0, 1)
            elif depth_pos_embed.dim() == 3:
                if depth_pos_embed.shape[0] == tgt.shape[0]:
                    depth_memory = depth_pos_embed.transpose(0, 1)
                else:
                    depth_memory = depth_pos_embed
            else:
                raise ValueError("Unsupported depth_pos_embed shape")
        else:
            depth_memory = None

        if mask_depth is not None and mask_depth.dim() == 3:
            mask_depth = mask_depth.flatten(1)

        # 1) Depth Cross Attention
        # MHA input should be [L, B, C]
        # --------------------------------------------------
        if depth_memory is not None:
            q = self.with_pos_embed(tgt, query_pos).transpose(0, 1)

            tgt2 = self.cross_attn_depth(
                query=q,
                key=depth_memory,
                value=depth_memory,
                key_padding_mask=mask_depth
            )[0].transpose(0, 1)

            tgt = tgt + self.dropout_depth(tgt2)
            tgt = self.norm_depth(tgt)

        # --------------------------------------------------
        # 2) Self Attention
        # --------------------------------------------------
        q = k = self.with_pos_embed(tgt, query_pos)  # [B, Q, C]
        q = q.transpose(0, 1)  # [Q, B, C]
        k = k.transpose(0, 1)  # [Q, B, C]
        v = tgt.transpose(0, 1)  # [Q, B, C]

        tgt2 = self.self_attn(
            query=q,
            key=k,
            value=v
        )[0].transpose(0, 1)  # [B, Q, C]

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # --------------------------------------------------
        # 3) Text Cross Attention (optional)
        # text_memory in D2D is typically [B, T, C]
        # --------------------------------------------------
        if text_memory is not None:
            q = self.with_pos_embed(tgt, query_pos).transpose(0, 1)  # [Q, B, C]

            if text_memory.dim() == 3 and text_memory.shape[0] == tgt.shape[0]:
                k = text_memory.transpose(0, 1)  # [T, B, C]
                v = text_memory.transpose(0, 1)
            else:
                k = text_memory
                v = text_memory

            tgt2 = self.cross_attn_text(
                query=q,
                key=k,
                value=v,
                key_padding_mask=text_mask
            )[0].transpose(0, 1)  # [B, Q, C]

            tgt = tgt + self.dropout_text(tgt2)
            tgt = self.norm_text(tgt)

        # --------------------------------------------------
        # 4) Visual Cross Attention
        # In D2D, MSDeformAttn uses [B, Q, C], so NO transpose here
        # --------------------------------------------------
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),  # [B, Q, C]
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
            src_padding_mask
        )

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # --------------------------------------------------
        # 5) FFN
        # --------------------------------------------------
        tgt = self.forward_ffn(tgt)

        return tgt

class DepthAwareDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.dim_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, depth_pos_embed=None, mask_depth=None, text_memory=None, text_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):

            if reference_points.shape[-1] == 6:
                reference_points_input = reference_points[:, :, None] * \
                                         torch.cat([src_valid_ratios, src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            output = layer(output,
                           query_pos,
                           reference_points_input,
                           src,
                           src_spatial_shapes,
                           src_level_start_index,
                           src_padding_mask,
                           depth_pos_embed,
                           mask_depth,
                           text_memory=text_memory,
                           text_mask=text_mask)

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 6:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_det3d_transformer(cfg, text_guided_adapter=None):
    return D3DTransformer(
        d_model=cfg['hidden_dim'],
        dropout=cfg['dropout'],
        activation="relu",
        nhead=cfg['nheads'],
        dim_feedforward=cfg['dim_feedforward'],
        num_decoder_layers=cfg['dec_layers'],
        return_intermediate_dec=cfg['return_intermediate_dec'],
        num_feature_levels=cfg['num_feature_levels'],
        dec_n_points=cfg['dec_n_points'],
        text_guided_adapter=text_guided_adapter)
