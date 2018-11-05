from preprocessing import news_preprocess
from training_data_generation import Preprocess
from train import train


data_dir = '../Data/'
window = 1
stride = 1
begin_date = 20150901
end_date = 20150904

#------------------ preprocessing --------------------

prepro = news_preprocess()

prepro.pre_process(data_dir + 'signal.csv', 'content', 'published', begin = begin_date, end = end_date)

prepro.cut_and_slide(data_dir, window, stride)


#------------------ building training data --------------------
for i in range(3):
	inp_data_path = '../Data/articles_preprocessed_' + str(i) + '.pickle'
	res_data_path = '../Data/training_data/period'+str(i)+'/'
	preprocess = Preprocess(window=5, data_dir=res_data_path)
	preprocess.build(inp_data_path, 30000)
	preprocess.convert(inp_data_path)

#------------------ adjacent training --------------------

# global vars
e_dim = 300
n_negs = 40
epoch = 3
mb = 1024
ss_t = 1e-5
conti = False
weights = True
cuda = True

for i in range(4):
	name = 'period' + str(i)
	data_dir_1 = '/home/paperspace/projects/news_exploration/Data/training_data/period' + str(i) + '/'
	if i > 0:
		data_dir_0 = '/home/paperspace/projects/news_exploration/Data/training_data/period' + str(i-1) + '/'
	else:
		data_dir_0 = None
	train(name = name, data_dir_1 = data_dir_1, save_dir = data_dir_1,
		e_dim = e_dim, n_negs = n_negs, epoch = epoch, mb = mb,
		ss_t = ss_t, conti = conti, weights = weights, cuda = cuda, data_dir_0 = data_dir_0)
	print('adjacent training terminated')