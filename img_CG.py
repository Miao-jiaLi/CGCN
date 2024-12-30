import pdb
import sys

import numpy as np
import copy
from openvqa.models.mcan.adapter import Adapter
import torch.nn as nn
from P_C.PC import pc
import time
import torch
from openvqa.utils.make_mask import make_mask

class img_Net(nn.Module):
    def __init__(self, __C):
        super(img_Net, self).__init__()

        self.adapter = Adapter(__C)

    def forward(self, frcn_feat, grid_feat, bbox_feat):

        # lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        # lang_feat = self.embedding(ques_ix)
        # lang_feat, _ = self.lstm(lang_feat)

        img_feat, img_feat_mask = self.adapter(frcn_feat, grid_feat, bbox_feat)
        # pdb.set_trace()
        start = time.time()
        # wq = np.zeros((lang_feat.shape[0], 14, 14))
        # lang_feat_cpu = lang_feat.cpu()
        # for i in range(lang_feat.shape[0]):
        #     feat_len = torch.sum(torch.count_nonzero(lang_feat[i], dim=1) > 0)
        #     lang = lang_feat_cpu[i][0:feat_len].numpy()
        #     # lang = lang_feat_cpu[i].numpy()
        #     lang = np.transpose(lang)
        #     pDAG, _ = pc(lang, 0.05)
        #     # pDAG = torch.from_numpy(pDAG).cuda()
        #     wq[i][0:feat_len, 0:feat_len] = wq[i][0:feat_len, 0:feat_len] + pDAG
        #     sys.stdout.write(f"\rques_NO.{i}")
        #     sys.stdout.flush()

        wx = np.zeros((img_feat.shape[0], 100, 100))
        img_feat_cpu = img_feat.cpu()
        processed_feat = {}
        # pdb.set_trace()
        for j in range(img_feat.shape[0]):

            feat_len = torch.sum(img_feat_mask[j] == False).item()
            img = img_feat_cpu[j][0:feat_len].numpy()
            # img_tuple = tuple(img.flatten())
            img_str = np.array2string(img, separator=',')
            if img_str in processed_feat:
                wx[j] = wx[processed_feat[img_str]]
            else:
                img = np.transpose(img)
                pDAG, _ = pc(img, 0.05)
                wx[j][0:feat_len, 0:feat_len] = wx[j][0:feat_len, 0:feat_len] + pDAG
                # pdb.set_trace()
                processed_feat[img_str] = j
            sys.stdout.write(f"\r,img_NO{j}")
            sys.stdout.flush()
        end = time.time()  # 记录程序结束时间
        print(",程序执行时间为：", end - start, "秒")  # 输出程序执行时间
        # pdb.set_trace()
        return wx