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


class news_preprocessing:

	def __init__(self, window = 30, period = 7, cores = 1):
		self.window = window
		self.period = period
		self.cores = cores
		print('The window size is ', str(self.window), ' days')
		print('The periodical progression size is ', str(self.window), ' days')

	def pre_process(self, df, content_col, timestamp_col, begin = None, end = None):
		if set([isinstance(day, datetime.datetime) for day in df[timestamp_col]]) != {True}:
			raise Exception('timestamp_col contains values that are not of the datetime type')

		#trimming df
		df = self.trim_df(df, begin, end, content_col, timestamp_col)

		#parallelised tokenisation
		if self.core == 1:
			df[content_col] = [[str(word) for word in list(tk) if re.match('[\W_]+$', str(word)) is not None] for tk in tqdm(nlp.tokenizer(df[content_col]))]
		else:
			p = Pool(self.core, maxtasksperchild = 1)
			toks = p.map(nlp.tokenizer, tqdm(df[content_col]))
			p.close()
			df[content_col] = [[str(word) for word in list(tk) if re.match('[\W_]+$', str(word)) is not None] for tk in tqdm(tks)]

		self.df = df

	def trim_df(df, begin, end, content_col, timestamp_col):
		df = df[content_col, timestamp_col]
		if begin == end == None:
			return(df)
		elif begin == None:
			return(df[[day <= end for day in df[timestamp_col]]])
		elif end == None:
			return(df[[begin <= day for day in df[timestamp_col]]])
		else:
			return(df[[begin <= day <= end for day in df[timestamp_col]]])

	def

