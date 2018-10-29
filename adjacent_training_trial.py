import numpy as np
import pickle
import os
from train import train
from model import SGNS


name = 'period1'
data_dir_1 = '/home/paperspace/projects/news_exploration/Data/training_data/period1/'
data_dir_0 = '/home/paperspace/projects/news_exploration/Data/training_data/period0/'
save_dir = data_dir_1
e_dim = 300
n_negs = 40
epoch = 5
mb = 1024
ss_t = 1e-5
conti = False
weights = True
cuda = True

train(name = name, data_dir_1 = data_dir_0, save_dir = data_dir_0,
      e_dim = e_dim, n_negs = n_negs, epoch = epoch, mb = mb,
      ss_t = ss_t, conti = conti, weights = weights, cuda = cuda, data_dir_0 = None)

print('first period training terminated')


train(name = name, data_dir_1 = data_dir_1, save_dir = save_dir,
      e_dim = e_dim, n_negs = n_negs, epoch = epoch, mb = mb,
      ss_t = ss_t, conti = conti, weights = weights, cuda = cuda, data_dir_0 = data_dir_0)

print('adjacent training terminated')