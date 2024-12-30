# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------
import pdb

from openvqa.utils.make_mask import make_mask
from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm
from openvqa.models.CGCN.mca import MCA_ED
from openvqa.models.CGCN.adapter import Adapter

import torch.nn as nn
import torch.nn.functional as F
import torch


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.__C = __C

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.adapter = Adapter(__C)

        self.backbone = MCA_ED(__C)
        # imgfeat_linear_size = __C.FEAT_SIZE['gqa']['FRCN_FEAT_SIZE'][1]
        # Flatten to vector
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)
        # self.frcn_linear = nn.Linear(imgfeat_linear_size, __C.HIDDEN_SIZE)
        # Classification layers
        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj_norm1 = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj_norm2 = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)
        self.proja = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)
        self.projb = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)


    # def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix, img_graph):
    def forward(self, frcn_feat, gird_feat, bbox_feat, img_graph, ques_ix):

        # Pre-process Language Feature
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)

        lang_feat, _ = self.lstm(lang_feat)
        # img_feat = self.frcn_linear(frcn_feat)
        img_feat, img_feat_mask = self.adapter(frcn_feat, gird_feat, bbox_feat)

        # if img_feat.shape[1] > 100:
        #     img_graph_exp = torch.ones((img_feat.shape[0], img_feat.shape[1], img_feat.shape[1]))
        #     img_graph_exp[:, 0:100, 0:100] = img_graph
        #     img_graph = img_graph_exp.to(img_graph.device)
        # img_feat_mask = make_mask(img_feat)
        # pdb.set_trace()
        # Backbone Framework
        lang_feat, img_feat_C, img_feat_N = self.backbone(
            lang_feat,
            img_feat,
            img_graph,
            lang_feat_mask,
            img_feat_mask
        )

        # Flatten to vector
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat_C = self.attflat_img(
            img_feat_C,
            img_feat_mask
        )

        img_feat_N = self.attflat_img(
            img_feat_N,
            img_feat_mask
        )
        # Classification layers
        proj_feat_a = lang_feat + img_feat_C
        proj_feat_b = lang_feat + img_feat_N

        proj_feat_a = self.proj_norm(proj_feat_a)
        proj_feat_b = self.proj_norm1(proj_feat_b)
        proj_feat = self.proj_norm2(proj_feat_a + proj_feat_b)
        # proj_feat = self.proj(proj_feat)

        proj_feat_a = self.proja(proj_feat_a)
        proj_feat_b = self.projb(proj_feat_b)
        proj_feat = self.proj(proj_feat)

        return proj_feat, proj_feat_a, proj_feat_b

