# -*- coding: utf-8 -*-

import os
import pickle
import random
import torch as t
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from model import Word2Vec, SGNS

# ------------------------------ Variable Definition ------------------------------
# name: name of result file to be saved
# data_dir_1: directory to the current period's data
# save_dir: where to save the result
# e_dim: number of dimensions of the embeddings
# n_negs: number of negative words to sample
# epoch: number of epochs
# mb: batch size
# ss_t: threshold used for subsampling on the vocab
# conti: boolean, whether we're starting from scratch or continuing from a previous training session
# weights: boolean, whether to use weights for negative sampling. If True, then we use the unigram distribution raised to the power of 3/4
# cuda: whether to use GPU for training, the current version only works with cude = True
# data_dir_0: where the previous period's data is stored, 'None' by default


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


def train(name, data_dir_1, save_dir, e_dim, n_negs, epoch, mb, ss_t, conti, weights, cuda = True, data_dir_0 = None):

    idx2word_1 = pickle.load(open(os.path.join(data_dir_1, 'idx2word.dat'), 'rb'))
    word2idx_1 = pickle.load(open(os.path.join(data_dir_1, 'word2idx.dat'), 'rb'))

    #creating idx2idx dict for the overlapping section of the vocabularies
    if data_dir_0 is not None:
        word2idx_0 = pickle.load(open(os.path.join(data_dir_0, 'word2idx.dat'), 'rb'))
        vocab_inters = set(word2idx_0.keys())&set(word2idx_1.keys())
        idx2idx = {word2idx_1[word]: word2idx_0[word] for word in vocab_inters}
        if data_dir_0 is not None:
            with open(data_dir_0 +'idx2vec.dat', 'rb') as handle:
                previous_model = pickle.load(handle)
    else:
        previous_model = None

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
            if data_dir_0 is not None:
                # here we need to create a idx2idx dict
                vocab_present = list(set(iword.cpu().numpy())&set(idx2idx.keys()))
                if len(vocab_present) != 0:
                    rwords_dict = {word:idx2idx[word] for word in vocab_present}
                else:
                    rwords_dict = None
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