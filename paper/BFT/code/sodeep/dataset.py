"""
This file contains functions for synthesizing data of various distributions.

It builds D_synthetic of Eq. (4) in the paper "Backpropagation-Free Test-Time
Adaptation for Lightweight EEG-Based BCIs" (IEEE J. Biomed. Health Inform.,
2026): random score vectors x_tilde in R^K together with their ground-truth
rank vectors pi_tilde, used to pre-train the mapping module m.
"""

import numpy as np
import torch

from random import randint
from torch.utils.data import Dataset


def get_rand_seq(seq_len, ind=None):
    if ind is None:
        type_rand = randint(0, 9)
    else:
        type_rand = int(ind)

    if type_rand == 0:
        rand_seq = np.random.rand(seq_len) * 2.0 - 1
    elif type_rand == 1:
        rand_seq = np.random.uniform(-1, 1, seq_len)
    elif type_rand == 2:
        rand_seq = np.random.standard_normal(seq_len)
    elif type_rand == 3:
        a = np.random.rand()
        b = np.random.rand()
        rand_seq = np.arange(a, b, (b - a) / seq_len)
    elif type_rand == 4:
        a = np.random.rand()
        b = np.random.rand()
        rand_seq = np.arange(a, b, (b - a) / seq_len)
        np.random.shuffle(rand_seq)
    elif type_rand == 5:
        split = randint(1, seq_len)
        rand_seq = np.concatenate(
            [np.random.rand(split) * 2.0 - 1, np.random.standard_normal(seq_len - split)])
    elif type_rand == 6:
        split = randint(1, seq_len)
        rand_seq = np.concatenate(
            [np.random.uniform(-1, 1, split), np.random.standard_normal(seq_len - split)])
    elif type_rand == 7:
        split = randint(1, seq_len)
        rand_seq = np.concatenate(
            [np.random.rand(split) * 2.0 - 1, np.random.uniform(-1, 1, seq_len - split)])
    elif type_rand == 8:
        split = randint(1, seq_len)
        a = np.random.rand()
        b = np.random.rand()
        rand_seq = np.arange(a, b, (b - a) / split)
        np.random.shuffle(rand_seq)
        rand_seq = np.concatenate(
            [rand_seq, np.random.rand(seq_len - split) * 2.0 - 1])
    elif type_rand == 9:
        a = -1.0
        b = 1.0
        rand_seq = np.arange(a, b, (b - a) / seq_len)
    elif type_rand == 10:
        rand_seq = np.random.rand(seq_len) * np.random.rand() - np.random.rand()
    elif type_rand == 11:
        rand_seq = np.random.rand(seq_len)

    return rand_seq[:seq_len]


class SeqDataset(Dataset):

    def __init__(self, seq_len, nb_sample=400000, dist=None):
        self.seq_len = seq_len
        self.nb_sample = nb_sample

        self.dist = dist

    def __getitem__(self, index):
        # one synthetic pair (x_tilde, pi_tilde) of Eq. (4).  dist=11 draws
        # x_tilde uniformly from [0, 1], as described in Section III-C.
        rand_seq = get_rand_seq(self.seq_len, self.dist)
        zipp_sort_ind = zip(np.argsort(rand_seq)[::-1], range(self.seq_len))

        # NOTE: the ranks are normalised to {1/K, 2/K, ..., 1} with the LARGEST
        # score receiving 1.  Eq. (4) writes pi_tilde in {1, 2, ..., K}; the two
        # differ by the constant factor K, which the L1 objective absorbs.
        ranks = [((y[1] + 1) / float(self.seq_len)) for y in sorted(zipp_sort_ind, key=lambda x: x[0])]

        return torch.FloatTensor(rand_seq), torch.FloatTensor(ranks)

    def __len__(self):
        return self.nb_sample


def get_rank_single(batch_score):
        rank = torch.argsort(batch_score, dim=0)
        rank = torch.argsort(rank, dim=0)
        rank = (rank * -1) + batch_score.size(0)
        rank = rank.float()
        rank = rank / batch_score.size(0)

        return rank
