# -*- coding: utf-8 -*-

import os
import pickle
import argparse
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--vocab', type=str, default='./data/corpus.txt', help="corpus path for building vocab")
    parser.add_argument('--corpus', type=str, default='./data/corpus.txt', help="corpus path")
    parser.add_argument('--unk', type=str, default='<UNK>', help="UNK token")
    parser.add_argument('--window', type=int, default=5, help="window size")
    parser.add_argument('--max_vocab', type=int, default=20000, help="maximum number of vocab")
    return parser.parse_args()


class Preprocess(object):

    def __init__(self, window=5, unk='<UNK>', data_dir='./data/'):
        self.window = window
        self.unk = unk
        self.data_dir = data_dir

    def skipgram(self, sentence, i):
        iword = sentence[i]
        left = sentence[max(i - self.window, 0): i]
        right = sentence[i + 1: i + 1 + self.window]

        # <UNK> below here is used to indicate padding
        return(iword, [self.unk for _ in range(self.window - len(left))] + left + right + [self.unk for _ in range(self.window - len(right))])

    def build(self, filepath, max_vocab=20000):
        print("building vocab...")
        step = 0
        self.wc = {self.unk: 1}
        with open(filepath, 'rb') as handle:
            file = pickle.load(handle)
            for line in file:
                step += 1
                # count till step == 1000 (when step = 1000, step % 1000 = 0 and not step % 1000 == True)
                if not step % 1000:
                    print("working on {}kth line".format(step // 1000), end='\r')
                for word in line:
                    #the below function updates the word count in the self.wc dictionary
                    #the get function is good, avoid writting multiple lines
                    self.wc[word] = self.wc.get(word, 0) + 1
        print("")
        # creates the idx2word list with the option to curb the vocab length with the variable max_vocab
        # The vocab list is sorted by descending order of the word count
        self.idx2word = [self.unk] + sorted(self.wc, key=self.wc.get, reverse=True)[:max_vocab - 1]
        # creating dict that allows to find the index of a particular word
        self.word2idx = {self.idx2word[idx]: idx for idx, _ in enumerate(self.idx2word)}
        # Creating the vocab list
        self.vocab = set([word for word in self.word2idx])
        pickle.dump(self.wc, open(os.path.join(self.data_dir, 'wc.dat'), 'wb'))
        pickle.dump(self.vocab, open(os.path.join(self.data_dir, 'vocab.dat'), 'wb'))
        pickle.dump(self.idx2word, open(os.path.join(self.data_dir, 'idx2word.dat'), 'wb'))
        pickle.dump(self.word2idx, open(os.path.join(self.data_dir, 'word2idx.dat'), 'wb'))
        print("build done")

    def convert(self, filepath):
        print("converting corpus...")
        step = 0
        data = []
        with open(filepath, 'rb') as handle:
            file = pickle.load(handle)
            for line in file:
                step += 1
                if not step % 1000:
                    print("working on {}kth line".format(step // 1000), end='\r')
                sent = []
                for word in line:
                    if word in self.vocab:
                        sent.append(word)
                    else:
                        sent.append(self.unk)
                for i in range(len(sent)):
                    iword, owords = self.skipgram(sent, i)
                    data.append((self.word2idx[iword], [self.word2idx[oword] for oword in owords]))
        print("")
        pickle.dump(data, open(os.path.join(self.data_dir, 'train.dat'), 'wb'))
        print("conversion done")


if __name__ == '__main__':
    args = parse_args()
    preprocess = Preprocess(window=args.window, unk=args.unk, data_dir=args.data_dir)
    preprocess.build(args.vocab, max_vocab=args.max_vocab)
    preprocess.convert(args.corpus)