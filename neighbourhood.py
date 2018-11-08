import pickle
import numpy as np
from scipy import spatial

class similar_words:

	def __init__(self, data_dir):
		with open(data_dir + 'word2idx.dat', 'rb') as handle:
			self.word2idx = pickle.load(handle)
		with open(data_dir + 'idx2word.dat', 'rb') as handle:
			self.idx2word = pickle.load(handle)
		with open(data_dir + 'idx2vec.dat', 'rb') as handle:
			self.idx2vec = pickle.load(handle)

	def normalise(self):
		l2_norm = np.sqrt((self.idx2vec * self.idx2vec).sum(axis = 1)).reshape((self.idx2vec.shape[0]),1)
		self.n_idx2vec = self.idx2vec/l2_norm
		self.n_idx2vec[0] = np.array([0]*self.idx2vec.shape[1])


	def closest_neighbours(self, word, n = 10, normalised = True):
		idx = self.word2idx[word]
		if normalised:
			idx_sim_dict = {i:1-spatial.distance.cosine(self.n_idx2vec[idx],vec) for i,vec in enumerate(self.n_idx2vec) if i not in [0,idx]}
		else:
			idx_sim_dict = {i:1-spatial.distance.cosine(self.idx2vec[idx],vec) for i,vec in enumerate(self.idx2vec) if i not in [0,idx]}
		idx_sim_dict = sorted(idx_sim_dict.items(), key=lambda kv: kv[1], reverse=True)
		self.neighbours = [(self.idx2word[pair[0]], pair[1]) for pair in idx_sim_dict[:n]]