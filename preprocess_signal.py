from preprocessing import news_preprocess
from training_data_generation import Preprocess


data_dir = '../Data/'
window = 7
stride = 1
begin_date = 20150901
end_date = 20151001

#------------------ preprocessing --------------------

prepro = news_preprocess(cores = 8)

prepro.pre_process(data_dir + 'signal.csv', 'content', 'published', begin = begin_date, end = end_date)

prepro.cut_and_slide(window,stride)

prepro.save_to_pickle(data_path+'signal_')


#------------------ building training date --------------------
# inp_data_path = '../Data/all-the-news/articles_1_0_preprocessed.pickle'
# res_data_path = '../Data/training_data/period0/'
