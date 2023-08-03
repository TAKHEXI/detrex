# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import torch
import torch.nn as nn
import torchvision

from detrex.layers import (
    FFN,
    MLP,
    BaseTransformerLayer,
    MultiheadAttention,
    MultiScaleDeformableAttention,
    TransformerLayerSequence,
    get_sine_pos_embed,
)
from detrex.utils import inverse_sigmoid
from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

class DabDeformableDetrTransformerEncoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        operation_order: tuple = ("self_attn", "norm", "ffn", "norm"),
        num_layers: int = 6,
        post_norm: bool = False,
        num_feature_levels: int = 4,
    ):
        super(DabDeformableDetrTransformerEncoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=MultiScaleDeformableAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=attn_dropout,
                    batch_first=True,
                    num_levels=num_feature_levels,
                ),
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    output_dim=embed_dim,
                    num_fcs=2,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(embed_dim),
                operation_order=operation_order,
            ),
            num_layers=num_layers,
        )
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):

        for i, layer in enumerate(self.layers):
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                layer_index=i,
                **kwargs,
            )

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class DabDeformableDetrTransformerDecoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        num_layers: int = 6,
        return_intermediate: bool = True,
        num_feature_levels: int = 4,
    ):
        super(DabDeformableDetrTransformerDecoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=[
                    MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=True,
                    ),
                    MultiScaleDeformableAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        dropout=attn_dropout,
                        batch_first=True,
                        num_levels=num_feature_levels,
                    ),
                ],
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    output_dim=embed_dim,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(embed_dim),
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.return_intermediate = return_intermediate

        self.query_scale = MLP(embed_dim, embed_dim, embed_dim, 2)
        self.ref_point_head = MLP(2 * embed_dim, embed_dim, embed_dim, 2)

        self.bbox_embed = None
        self.class_embed = None

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        reference_points=None,  # num_queries, 4. normalized.
        valid_ratios=None,
        **kwargs,
    ):
        output = query
        bs, num_queries, _ = output.size()
        if reference_points.dim() == 2:
            reference_points = reference_points.unsqueeze(0).repeat(bs, 1, 1)  # bs, num_queries, 4

        intermediate = []
        intermediate_reference_points = []
        for layer_idx, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
            raw_query_pos = self.ref_point_head(query_sine_embed)
            pos_scale = self.query_scale(output) if layer_idx != 0 else 1
            query_pos = pos_scale * raw_query_pos

            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                query_sine_embed=query_sine_embed,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points_input,
                **kwargs,
            )

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_idx](output)
                if reference_points.shape[-1] == 4:
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


class DabDeformableDetrTransformer(nn.Module):
    """Transformer module for DAB-Deformable-DETR

    Args:
        encoder (nn.Module): encoder module.
        decoder (nn.Module): decoder module.
        as_two_stage (bool): whether to use two-stage transformer. Default False.
        num_feature_levels (int): number of feature levels. Default 4.
        two_stage_num_proposals (int): number of proposals in two-stage transformer. Default 100.
            Only used when as_two_stage is True.
    """

    def __init__(
        self,
        encoder=None,
        decoder=None,
        as_two_stage=False,
        num_feature_levels=4,
        two_stage_num_proposals=300,
        topK=20,
    ):
        super(DabDeformableDetrTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals

        self.embed_dim = self.encoder.embed_dim

        # scale-level embedding
        # 对 4 个多尺度特征层每层附加一个 256 维的 embedding
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dim))

        # if self.as_two_stage:
        self.enc_output = nn.Linear(self.embed_dim, self.embed_dim)
        self.enc_output_norm = nn.LayerNorm(self.embed_dim)
        self.topK = topK
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.normal_(self.level_embeds)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N, S, C = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H * W)].view(N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            proposals.append(proposal)
            _cur += H * W

        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(
            -1, keepdim=True
        )
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float("inf")
        )
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in encoder and decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The ratios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            )
            # 将各层特征图每个特征点中心坐标根据特征图非 padding 的边长进行归一化
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)   # (1, HW) / (bs, 1)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)   # (1, HW) / (bs, 1)
            ref = torch.stack((ref_x, ref_y), -1)                                   # (bs, HW, 2)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid ratios of feature maps of all levels."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def nms(self, bboxes, scores, threshold=0.5):
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)  # [N,] 每个bbox的面积
        _, order = scores.sort(0, descending=True)  # 降序排列

        keep = []
        while order.numel() > 0:  # torch.numel()返回张量元素个数
            if order.numel() == 1:  # 保留框只剩一个
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()  # 保留scores最大的那个框box[i]
                keep.append(i)

            # 计算box[i]与其余各框的IOU
            xx1 = x1[order[1:]].clamp(min=x1[i])  # [N-1,]
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])
            inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)  # [N-1,]

            iou = inter / (areas[i] + areas[order[1:]] - inter)  # [N-1,]
            idx = (iou <= threshold).nonzero().squeeze()  # 注意此时idx为[N-1,] 而order为[N,]
            if idx.numel() == 0:
                break
            order = order[idx + 1]  # 修补索引之间的差值
        return torch.LongTensor(keep)  # Pytorch的索引值为LongTensor


    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        query_embed,
        WH,                         # (bs, 4) 第二维是图片预处理后的 [W, H, W, H]
        **kwargs,
    ):
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
            zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)
        ):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            feat = feat.flatten(2).transpose(1, 2)  # 从 (bs, c, h, w) 变为 (bs, hw, c)
            mask = mask.flatten(1)                  # 从 (bs, h, w) 变为 (bs, hw)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # 从 (bs, c, h, w) 变为 (bs, hw, c)
            # pos_embed 只能够区分 h,w 位置，无法区分不同特征层上坐标相同的特征点，附加 scale-level embedding 信息，区分所有特征点
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1) # (bs, hw, c) + (1, 1, 256) , c=256
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)

        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device
        )

        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,  # bs, num_token, num_level, 2
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )

        bs, _, c = memory.shape
        # if self.as_two_stage:
        # assert query_embed is None, "query_embed should be None in two-stage"
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )
        # output_memory: bs, num_tokens, c
        # output_proposals: bs, num_tokens, 4. unsigmoided.
        # output_proposals: bs, num_tokens, 4

        # 对encoder得到的编码特征memory进行分类预测，对应proposals的分类结果（多分类）
        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory) # (bs, sum(spatial_shapes[i,0]*spatial_shapes[i,1]), c=80)

        # 对encoder得到的编码特征memory进行回归预测，得到相对于proposals(xywh)的偏移量，然后和proposals相加，得到预测的proposals
        enc_outputs_coord_unact = (
            self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
        )  # unsigmoided.  (bs, sum(spatial_shapes[i,0]*spatial_shapes[i,1]), 4)

        # TODO: NMS 去重 (先去重再取 topK)

        # 取 topK 的预测 proposals 作为 decoder 的输入（这里是取了类别可能性最大的 topK 个）
        # topk = self.two_stage_num_proposals
        topk = self.topK
        # max(-1) 按 enc_outputs_class 的最后一维 （80）取最大，[0] 返回最大值的每个数
        # torch.topK()[1] 返回元素在原数组中的下标, dim=1 在第一维上排序
        topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]
        # 预测 proposals 对应的类别
        topK_class = torch.topk(enc_outputs_class.max(-1)[1], topk, dim=1)[0]

        # extract region proposal boxes 取出 topK 得分最高对应的预测 bbox (bs, topK, 4)
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        )  # unsigmoided.

        # 取消梯度，归一化，在2-stage中这个结果会送到 decoder 中作为初始的 bbox
        reference_points_2stage = topk_coords_unact.detach().sigmoid()

        # TODO: NMS 去重 (先取 topK 再去重)
        reference_points_2stage_afternms = []
        scores = enc_outputs_class.max(-1)[0]  # (bs, sum(spatial_shapes[i,0]*spatial_shapes[i,1]))
        topK_scores = torch.gather(
            scores, 1, topk_proposals
        )
        for b in range(bs):
            score = topK_scores[b]
            bbox = reference_points_2stage[b]
            wh = WH[b]
            boxes_xyxy = np.array([[(int)(x * y) for x, y in zip(box_cxcywh_to_xyxy(box).cpu().numpy(), wh)] for box in bbox])
            boxes_xyxy = torch.from_numpy(boxes_xyxy).cuda()
            idx = self.nms(boxes_xyxy, score, 0.5).cuda()
            reference_points_2stage_nms = torch.index_select(reference_points_2stage[b], 0, idx)
            reference_points_2stage_afternms.append(reference_points_2stage_nms)

        ref_length = len(reference_points_2stage_afternms[0])
        for row in reference_points_2stage_afternms:
            ref_length = min(ref_length, len(row))
        reference_points_2stage_afternms = [row[:ref_length] for row in reference_points_2stage_afternms]
        reference_points_2stage_afternms = torch.stack(reference_points_2stage_afternms)
        init_reference_out_2stage = reference_points_2stage_afternms

        # extract region features 生成decoder对应的query(target)
        target_unact = torch.gather(
            output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
        )
        target_2stage = target_unact.detach()

        # 1-stage
        reference_points = query_embed[..., self.embed_dim :].sigmoid()
        target = query_embed[..., : self.embed_dim]
        target = target.unsqueeze(0).expand(bs, -1, -1)
        init_reference_out = reference_points
        # (300, 4)

        # decoder
        inter_states, inter_references = self.decoder(
            query=target,  # bs, num_queries, embed_dims
            key=memory,  # bs, num_tokens, embed_dims
            value=memory,  # bs, num_tokens, embed_dims
            query_pos=None,
            key_padding_mask=mask_flatten,  # bs, num_tokens
            reference_points=reference_points,  # num_queries, 4
            spatial_shapes=spatial_shapes,  # nlvl, 2
            level_start_index=level_start_index,  # nlvl
            valid_ratios=valid_ratios,  # bs, nlvl, 2
            **kwargs,
        )

        inter_references_out = inter_references
        if self.as_two_stage:
            return (
                inter_states,
                init_reference_out,
                inter_references_out,
                target_unact,
                topk_coords_unact.sigmoid(),
            )
        return inter_states, init_reference_out, inter_references_out, None, None, reference_points_2stage_afternms
