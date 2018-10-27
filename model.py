# -*- coding: utf-8 -*-

import numpy as np
import torch as t
import torch.nn as nn
import pickle

from torch import LongTensor as LT
from torch import FloatTensor as FT

 
class Bundler(nn.Module):

    def forward(self, data):
        raise NotImplementedError

    def forward_i(self, data):
        raise NotImplementedError

    def forward_o(self, data):
        raise NotImplementedError


class Word2Vec(Bundler):

    def __init__(self, vocab_size=20000, embedding_size=300, padding_idx=0):
        # this instantiate the parent class
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        # define embeddings for the target word matrix and the context word matrix
        self.ivectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        # define parameters with initial weights: 0 for the padding and uniformly sampled weights for the vocab
        self.ivectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ovectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        # indicates that we do not exclude this as it is the main part of the graph - this should be treated as parameters
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True

    def load_idx2word(self, idx_dir):
        with open(idx_dir, 'rb') as handle:
            self.idx2word = pickle.load(idx_dir)


    def forward(self, data):
        return self.forward_i(data)

    def forward_i(self, data):
        v = LT(data)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        return self.ivectors(v)

    def forward_o(self, data):
        v = LT(data)
        v = v.cuda() if self.ovectors.weight.is_cuda else v
        return self.ovectors(v)


class SGNS(nn.Module):

    def __init__(self, embedding, vocab_size=20000, n_negs=20, weights=None, previous_model = None):
        super(SGNS, self).__init__()
        self.embedding = embedding
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        self.weights = None
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)
        self.previous_model = previous_model

    def forward(self, iword, owords, rwords_dict):
        batch_size = iword.size()[0]
        context_size = owords.size()[1]
        if self.weights is not None:
            # sampling negative words, per batch we have context_size * self.n_negs number of negative words
            nwords = t.multinomial(self.weights, batch_size * context_size * self.n_negs, replacement=True).view(batch_size, -1)
        else:
            nwords = FT(batch_size, context_size * self.n_negs).uniform_(0, self.vocab_size - 1).long()
        ivectors = self.embedding.forward_i(iword).unsqueeze(2)
        ovectors = self.embedding.forward_o(owords)
        nvectors = self.embedding.forward_o(nwords).neg()
        oloss = t.bmm(ovectors, ivectors).squeeze().sigmoid().log().mean(1)
        nloss = t.bmm(nvectors, ivectors).squeeze().sigmoid().log().view(-1, context_size, self.n_negs).sum(2).mean(1)

        rwords_dict = None if self.previous_model is None else rwords_dict
        
        if rwords_dict is not None:
            rwords = list(sorted(rwords_dict.keys()))
            print(rwords)
            print(rwords_dict[rwords[i]])
            rvectors = self.embedding.forward_i(rwords)
            MSE_loss_fun = nn.MSELoss(reduction = 'sum')
            total_r_loss = sum([MSE_loss_fun(rvectors[rvectors[i]], self.previous_model[rwords_dict[rwords[i]]]) for i in range(len(rwords))])
            return(-(oloss + nloss).mean() + 5*total_r_loss)
        else:
            return(-(oloss + nloss).mean())