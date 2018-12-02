from preprocessing import news_preprocess
from training_data_generation import Preprocess
from train import train

# Variables needed for preprocessing

# directory for data - this will be where we store generated data too
data_dir = '../Data/'
# the window size: how many consecutive days of text data do we capture at once
window = 1
# the number of days we advance per iteration
stride = 1
# the beginning date
begin_date = 20160621
# the end date
end_date = 20160627

#------------------ preprocessing --------------------

# instantiate
prepro = news_preprocess()

# preprocessing the data
prepro.pre_process(data_dir + 'all-the-news/articles_all.csv', 'content', 'date', begin = begin_date, end = end_date)

# cutting the data into the predefined intervals using a sliding window with the aforementioned stride size
prepro.cut_and_slide(data_dir = data_dir, window = window, period = stride)


#------------------ building training data --------------------
# there are 6 periods as a result of the preprocessing
for i in range(6):
	inp_data_path = '../Data/articles_preprocessed_' + str(i) + '.pickle'
	res_data_path = '../Data/training_data/period'+str(i)+'/'
	# instantiate
	preprocess = Preprocess(window=10, data_dir=res_data_path)
	# building the training data
	preprocess.build(inp_data_path, 30000)
	# converting to training data
	preprocess.convert(inp_data_path)

#------------------ adjacent training --------------------

# global vars
e_dim = 300
n_negs = 50
epoch = 40
mb = 1024
ss_t = 1e-5
conti = False
weights = True
cuda = True

for i in range(6):
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