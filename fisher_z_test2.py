#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2019/11/13 16:53
 @File    : fisher_z_test.py
 """

import torch
from scipy.stats import norm
import pdb


def get_partial_matrix(S, X, Y):
    S = S[X, :]
    S = S[:, Y]
    return S


def partial_corr_coef(S, i, j, Y):
    S = torch.tensor(S)
    X = [i, j]
    a = get_partial_matrix(S, Y, Y)
    det = torch.det(a)
    if det == 0:
        # print("矩阵不可逆")
        inv_syy = torch.pinverse(a)
    else:
        inv_syy = torch.inverse(a)

    i2 = 0
    j2 = 1
    # pdb.set_trace()
    # if torch.numel(inv_syy) == 0:
    #     S2 = get_partial_matrix(S, X, X)
    # else:
    #     # pdb.set_trace()
    S2 = get_partial_matrix(S, X, X) - get_partial_matrix(S, X, Y) @ inv_syy @ get_partial_matrix(S, Y, X)
    # pdb.set_trace()
    c = S2[i2, j2]
    r = c / torch.sqrt((S2[i2, i2] * S2[j2, j2]))

    return r


def cond_indep_fisher_z(data, var1, var2, cond=[], alpha=0.05):

    N, k_var = data.shape
    list_z = [var1, var2] + list(cond)
    list_new = []
    for a in list_z:
        list_new.append(int(a))
    data_array = torch.tensor(data)
    array_new = torch.transpose(torch.tensor(data_array[:, list_new]), 0, 1)
    cov_array = torch.cov(array_new)
    size_c = len(list_new)
    X1 = 0
    Y1 = 1
    S1 = [i for i in range(size_c) if i != 0 and i != 1]
    r = partial_corr_coef(cov_array, X1, Y1, S1)
    z = 0.5 * torch.log((1+r) / (1-r))
    z0 = 0
    # pdb.set_trace()
    W = torch.sqrt(torch.tensor(N - len(S1) - 3, dtype=torch.float)) * (z - z0)
    cutoff = norm.ppf(1 - 0.5 * alpha)
    if abs(W) < cutoff:
        CI = 1
    else:
        CI = 0
    #p = norm.cdf(W)
    r = abs(r)
    # r = None
    if torch.isnan(r):
        CI = None
    # pdb.set_trace()
    return CI, r






