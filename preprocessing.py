import pandas as pd
import re
import datetime
from tqdm import tqdm
import spacy
from multiprocessing import Pool
import pickle
import argparse
from gensim.test.utils import datapath
from gensim.models.word2vec import Text8Corpus
from gensim.models.phrases import Phrases, Phraser

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../Data/corpus.csv', help="input data directory path")
    parser.add_argument('--content_col', type=str, default='content', help="name of the column that contains the text content")
    parser.add_argument('--timestamp_col', type=str, default='date', help="name of the column that contains the dates")
    parser.add_argument('--cores', type=int, default=1, help="the number of cores used for multiprocessing")
    parser.add_argument('--begin', type=str, default='20000101', help="the starting date")
    parser.add_argument('--end', type=str, default='21000101', help="the ending date")
    parser.add_argument('--window', type=int, default=30, help="sliding window size")
    parser.add_argument('--period', type=int, default=7, help="sliding step size")
    parser.add_argument('--save_dir', type=str, default='../Data/', help="output data directory folder path, please finish with a /")
    return parser.parse_args()

class news_preprocess:

	def __init__(self):
		self.spacy = __import__('spacy')


	def pre_process(self, data_dir, content_col, timestamp_col, begin = None, end = None):
		self.timestamp_col = timestamp_col
		self.content_col = content_col

		#loading df and converting to datetime format
		df = pd.read_csv(data_dir)
		df[timestamp_col] = [datetime.datetime.strptime(day, '%Y-%m-%d') for day in df[timestamp_col]]

		if set([isinstance(day, datetime.datetime) for day in df[timestamp_col]]) != {True}:
			raise Exception('timestamp_col contains values that are not of the datetime type')

		#trimming df
		df = self.trim_df(df, content_col, timestamp_col, begin, end)
		df[content_col] = [re.sub(r'\W+', ' ', article).split(' ') for article in df[content_col]]

		#parallelised tokenisation
		#df[content_col] = [[str(word).lower().strip() for word in list(tk) if re.match('[\W_]+$', str(word)) is None] for tk in tqdm(self.nlp.tokenizer(df[content_col]))]
		df[content_col] = [[str(word).lower().strip() for word in list(tk) if re.match('[\W_]+$', str(word)) is None and len(str(word).lower().strip()) > 0] for tk in tqdm(df[content_col])]

		print('training for and detecting bigrams')
		phrases = phrases = Phrases(df[content_col], min_count=10, threshold=100)
		bigram = Phraser(phrases)
		df[content_col] = [bigram[article] for article in tqdm(df[content_col])]

		self.df = df
		print('the dataframe is preprocessed successfully')

	#trim the dataframe
	def trim_df(self, df, content_col, timestamp_col, begin, end):
		if not isinstance(begin, datetime.datetime) and not isinstance(end, datetime.datetime):
			begin = [datetime.datetime.strptime(str(date), '%Y%m%d') if date != None and not isinstance(date, datetime.datetime) else None for date in [begin, end]][0]
			end = [datetime.datetime.strptime(str(date), '%Y%m%d') if date != None and not isinstance(date, datetime.datetime) else None for date in [begin, end]][1]
		df = df[[content_col, timestamp_col]]
		if begin == end == None:
			return(df)
		elif begin == None:
			return(df[[day <= end for day in df[timestamp_col]]])
		elif end == None:
			return(df[[begin <= day for day in df[timestamp_col]]])
		else:
			return(df[[begin <= day <= end for day in df[timestamp_col]]])

	def cut_and_slide(self, window = 30, period = 7):

		min_date = min(self.df[self.timestamp_col])
		max_date = max(self.df[self.timestamp_col])

		begin_date = min_date
		#repartitioned_articles = []
		begin_dates = []

		for i in tqdm(range(int((int((max_date - min_date).days) - window) / period))):
			articles = list(self.df[[begin_date <= day < begin_date + datetime.timedelta(days = window) for day in self.df[self.timestamp_col]]][self.content_col])
			begin_dates.append(begin_date)
			begin_date += datetime.timedelta(days = 7)
			with open(path + 'preprocessed_' + str(i) + '.pickle', 'wb') as handle:
	 			pickle.dump(articles, handle, protocol = pickle.HIGHEST_PROTOCOL)
		self.begin_dates = begin_dates
		with open(path + 'begin_dates.pickle', 'wb') as handle:
			pickle.dump(self.begin_dates, handle, protocol = pickle.HIGHEST_PROTOCOL)
		#self.repartitioned_articles = repartitioned_articles
		print('articles repartitioned and saved')

	# def save_to_pickle(self, path):
	# 	with open(path + 'preprocessed.pickle', 'wb') as handle:
	# 		pickle.dump(self.repartitioned_articles, handle, protocol = pickle.HIGHEST_PROTOCOL)
	# 	with open(path + 'begin_dates.pickle', 'wb') as handle:
	# 		pickle.dump(self.begin_dates, handle, protocol = pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    args = parse_args()
    preprocess = news_preprocess(cores = args.cores)
    preprocess.preprocess(data_dir = argparse.data_dir, content_col = argparse.content_col, timestamp_col = argparse.timestamp_col, begin = argparse.begin, end = argparse.end)
    preprocess.cut_and_slide(window = argparse.window, period = argparse.period)
    preprocess.save_to_pickle(args.save_dir)

