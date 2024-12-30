# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------
import pdb

from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch
import math


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

class CMHAtt(nn.Module):
    def __init__(self, __C):
        super(CMHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask, graph):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)
        # pdb.set_trace()
        atted = self.att(v, k, q, mask, graph)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)
        # pdb.set_trace()
        return atted

    def att(self, value, key, query, mask, graph):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if graph is not None and mask is not None:
            identity_matrix = torch.eye(graph.shape[1]).to(graph.device)
            graph = torch.add(graph, identity_matrix)
            # pdb.set_trace()
            if scores.shape[2] > 100:
                img_graph_exp = torch.ones((graph.shape[0], scores.shape[2], scores.shape[2]))
                img_graph_exp[:, 0:100, 0:100] = graph
                graph = img_graph_exp.to(graph.device)

            graph_expanded = graph.unsqueeze(1)
            Causal_scores = torch.mul(scores, graph_expanded)
            Causal_scores = Causal_scores.to(torch.float32)
            # Causal_scores[Causal_scores == 0] = -1e9
            Causal_scores = Causal_scores.masked_fill(mask, -1e9)
            Causal_att_map = F.softmax(Causal_scores, dim=-1)

            # wight = Causal_att_map.cpu()
            # show_heatmaps(wight, xlabel='Keys', ylabel='Queries')
            att_map = self.dropout(Causal_att_map)
            # pdb.set_trace()
            return torch.matmul(att_map, value)
# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------
class NCMHAtt(nn.Module):
    def __init__(self, __C):
        super(NCMHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask, graph):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)
        # pdb.set_trace()
        atted = self.att(v, k, q, mask, graph)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)
        # pdb.set_trace()
        return atted

    def att(self, value, key, query, mask, graph):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if graph is not None and mask is not None:
            # identity_matrix = torch.eye(graph.shape[1]).to(graph.device)
            # pdb.set_trace()
            # graph = torch.add(graph, identity_matrix)
            if scores.shape[2] > 100:
                img_graph_exp = torch.zeros((graph.shape[0], scores.shape[2], scores.shape[2]))
                img_graph_exp[:, 0:100, 0:100] = graph
                graph = img_graph_exp.to(graph.device)
            graph_expanded = graph.unsqueeze(1)
            Causal_scores = torch.mul(scores, graph_expanded)
            Causal_scores = Causal_scores.to(torch.float32)
            scores = scores - Causal_scores
            # scores[scores == 0] = -1e9
            scores = scores.masked_fill(mask, -1e9)
            # pdb.set_trace()
            NonCausal_att_map = F.softmax(scores, dim=-1)
            # wight = NonCausal_att_map.cpu()
            # show_heatmaps(wight, xlabel='Keys', ylabel='Queries')
            att_map = self.dropout(NonCausal_att_map)


            return torch.matmul(att_map, value)
        # return torch.matmul(att_map, value)

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------
class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()

        self.mhatt1 = CMHAtt(__C)
        self.mhatt2 = MHAtt(__C)

        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)


    def forward(self, x, y, x_mask, y_mask, graphx):

        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask, graphx)
        ))
        # pdb.set_trace()
        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x
class SGNCA(nn.Module):
    def __init__(self, __C):
        super(SGNCA, self).__init__()

        # self.mhatt1 = CMHAtt(__C)
        self.mhatt1 = NCMHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        # self.mhatt4 = MHAtt(__C)

        # self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)


    def forward(self, x, y, x_mask, y_mask, graphx):
        # a = 0
        # x = self.norm1(x + self.dropout1(
        #     self.mhatt1(x, x, x, x_mask, graphx)
        # ))
        # cx = self.norm1(x + self.dropout1(
        #     self.mhatt1(x, x, x, x_mask, graphx)
        # ))
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask, graphx)
        ))
        # pdb.set_trace()

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))


        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))


        return x
# class SGA(nn.Module):
#     def __init__(self, __C):
#         super(SGA, self).__init__()
#
#         self.mhatt1 = MHAtt(__C)
#         self.mhatt2 = MHAtt(__C)
#         self.ffn = FFN(__C)
#
#         self.dropout1 = nn.Dropout(__C.DROPOUT_R)
#         self.norm1 = LayerNorm(__C.HIDDEN_SIZE)
#
#         self.dropout2 = nn.Dropout(__C.DROPOUT_R)
#         self.norm2 = LayerNorm(__C.HIDDEN_SIZE)
#
#         self.dropout3 = nn.Dropout(__C.DROPOUT_R)
#         self.norm3 = LayerNorm(__C.HIDDEN_SIZE)
#
#     def forward(self, x, y, x_mask, y_mask):
#         x = self.norm1(x + self.dropout1(
#             self.mhatt1(v=x, k=x, q=x, mask=x_mask)
#         ))
#
#         x = self.norm2(x + self.dropout2(
#             self.mhatt2(v=y, k=y, q=x, mask=y_mask)
#         ))
#
#         x = self.norm3(x + self.dropout3(
#             self.ffn(x)
#         ))
#
#         return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, __C):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])
        self.den_list = nn.ModuleList([SGNCA(__C) for _ in range(__C.LAYER)])

    def forward(self, x, y, gy, x_mask, y_mask):
        # def forward(self, x, y, x_mask, y_mask, c):
        # Get hidden vector
        cy = y.clone()
        ny = y.clone()
        # pdb.set_trace()
        gy = gy.squeeze()
        # pdb.set_trace()

        # pgy = gy.transpose(1, 2)
        # pcy = torch.add(gy, pgy)
        # pdb.set_trace()
        # CSy = CSy.squeeze()
        # pdb.set_trace()
        for enc in self.enc_list:
            x = enc(x, x_mask)
            # causal_x = enc(x, x_mask)
            # spurious_x = x - enc(x, x_mask)

        for dec in self.dec_list:
            # cy = y.clone()
            cy = dec(cy, x, y_mask, x_mask, gy)
            # pdb.set_trace()

        # pdb.set_trace()
        for den in self.den_list:
            # ny = y.clone()
            ny = den(ny, x, y_mask, x_mask, gy)
        # pdb.set_trace()

        # return x, cy
        return x, cy, ny
