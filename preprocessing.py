import pandas as pd
import numpy as np
import re
from collections import defaultdict
import datetime
from tqdm import tqdm
import spacy
from multiprocessing import Pool
import pickle
import string

class news_preprocess:

	def __init__(self, cores = 1):
		self.cores = cores
		self.spacy = __import__('spacy')


	def pre_process(self, df, content_col, timestamp_col, begin = None, end = None):
		self.timestamp_col = timestamp_col
		self.content_col = content_col

		if set([isinstance(day, datetime.datetime) for day in df[timestamp_col]]) != {True}:
			raise Exception('timestamp_col contains values that are not of the datetime type')

		#trimming df
		df = self.trim_df(df, content_col, timestamp_col, begin, end)

		#parallelised tokenisation
		if self.cores == 1:
			df[content_col] = [[str(word) for word in list(tk) if re.match('[\W_]+$', str(word)) is None] for tk in tqdm(self.nlp.tokenizer(df[content_col]))]
		else:
			nlp = self.spacy.load('en')
			p = Pool(self.cores, maxtasksperchild = 1)
			toks = p.map(nlp.tokenizer, tqdm(df[content_col]))
			p.close()
			df[content_col] = [[str(word) for word in list(tk) if re.match('[\W_]+$', str(word)) is None] for tk in tqdm(toks)]

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
		repartitioned_articles = []
		begin_dates = []

		for i in tqdm(range(int((int((max_date - min_date).days) - window) / period))):
			articles = self.df[[begin_date <= day < begin_date + datetime.timedelta(days = window) for day in self.df[self.timestamp_col]]][self.content_col]
			begin_dates.append(begin_date)
			begin_date += datetime.timedelta(days = 7)
			repartitioned_articles.append(articles)
		self.begin_dates = begin_dates
		self.repartitioned_articles = repartitioned_articles
		print('articles repartitioned, they can be accessed at self.reparitioned_articles')

	def save_to_pickle(self, path):
		with open(path, 'wb') as handle:
			pickle.dump(self.repartitioned_articles, handle, protocol = pickle.HIGHEST_PROTOCOL)



