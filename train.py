# -*- coding: utf-8 -*-

import os
import pickle
import random
# import argparse
import torch as t
import numpy as np

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from model import Word2Vec, SGNS


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--name', type=str, default='sgns', help="model name")
#     parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
#     parser.add_argument('--save_dir', type=str, default='./pts/', help="model directory path")
#     parser.add_argument('--e_dim', type=int, default=300, help="embedding dimension")
#     parser.add_argument('--n_negs', type=int, default=20, help="number of negative samples")
#     parser.add_argument('--epoch', type=int, default=100, help="number of epochs")
#     parser.add_argument('--mb', type=int, default=4096, help="mini-batch size")
#     parser.add_argument('--ss_t', type=float, default=1e-5, help="subsample threshold")
#     parser.add_argument('--conti', action='store_true', help="continue learning")
#     parser.add_argument('--weights', action='store_true', help="use weights for negative sampling")
#     parser.add_argument('--cuda', action='store_true', help="use CUDA")
#     return parser.parse_args()


class PermutedSubsampledCorpus(Dataset):
    # performs subsampling if desired

    def __init__(self, datapath, ws=None):
        data = pickle.load(open(datapath, 'rb'))
        if ws is not None:
            self.data = []
            for iword, owords in data:
                if random.random() > ws[iword]:
                    self.data.append((iword, owords))
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iword, owords = self.data[idx]
        return iword, np.array(owords)


def train(name, data_dir_1, save_dir, e_dim, n_negs, epoch, mb, ss_t, conti, weights, cuda, data_dir_0 = None):

    idx2word_1 = pickle.load(open(os.path.join(data_dir_1, 'idx2word.dat'), 'rb'))
    word2idx_1 = pickle.load(open(os.path.join(data_dir_1, 'word2idx.dat'), 'rb'))

    #creating idx2idx dict for the overlapping section of the vocabularies
    if data_dir_0 is not None:
        word2idx_0 = pickle.load(open(os.path.join(data_dir_0, 'word2idx.dat'), 'rb'))
        vocab_inters = set(word2idx_0.keys())&set(word2idx_1.keys())
        idx2idx = {word2idx_1[word]: word2idx_0[word] for word in vocab_inters}
        with open(data_dir_0 +'idx2vec.dat', 'rb') as handle:
            previous_model = pickle.load(handle)

    wc = pickle.load(open(os.path.join(data_dir_1, 'wc.dat'), 'rb'))
    wf = np.array([wc[word] for word in idx2word_1])
    wf = wf / wf.sum()
    ws = 1 - np.sqrt(ss_t / wf)
    ws = np.clip(ws, 0, 1)
    vocab_size = len(idx2word_1)
    weights = wf if weights else None
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    model = Word2Vec(vocab_size=vocab_size, embedding_size=e_dim)
    modelpath = os.path.join(save_dir, '{}.pt'.format(name))
    sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=n_negs, weights=weights, previous_model = previous_model)
    if os.path.isfile(modelpath) and conti:
        sgns.load_state_dict(t.load(modelpath))
    if cuda:
        sgns = sgns.cuda()
    optim = Adam(sgns.parameters())
    optimpath = os.path.join(save_dir, '{}.optim.pt'.format(name))
    if os.path.isfile(optimpath) and conti:
        optim.load_state_dict(t.load(optimpath))
    for epoch in range(1, epoch + 1):
        dataset = PermutedSubsampledCorpus(os.path.join(data_dir_1, 'train.dat'))
        #dataloader converts input numpy data into long tensors
        dataloader = DataLoader(dataset, batch_size=mb, shuffle=True)
        total_batches = int(np.ceil(len(dataset) / mb))
        pbar = tqdm(dataloader)
        pbar.set_description("[Epoch {}]".format(epoch))
        for iword, owords in pbar:
            # if cuda:
            #     iword = iword.cuda()
            #     owords = owords.cuda()
            vocab_present = list(set(iword.cpu().numpy())&set(idx2idx.keys()))
            if data_dir_0 is not None and len(vocab_present) != 0:
                # here we need to create a idx2idx dict
                rwords_dict = {word:idx2idx[word] for word in vocab_present}
            else:
                rwords_dict = None
            loss = sgns(iword, owords, rwords_dict)
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=loss.item())
    idx2vec = model.ivectors.weight.data.cpu().numpy()
    pickle.dump(idx2vec, open(os.path.join(data_dir_1, 'idx2vec.dat'), 'wb'))
    t.save(sgns.state_dict(), os.path.join(save_dir, '{}.pt'.format(name)))
    t.save(optim.state_dict(), os.path.join(save_dir, '{}.optim.pt'.format(name)))