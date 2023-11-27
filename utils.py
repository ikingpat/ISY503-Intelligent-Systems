import numpy as np

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
from collections import Counter
import string
import re
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def preprocess_string(s):
	# Remove all non-word characters (everything except numbers and letters)
	s = re.sub(r"[^\w\s]", '', s)
	# Replace all runs of whitespaces with no space
	s = re.sub(r"\s+", '', s)
	# replace digits with no space
	s = re.sub(r"\d", '', s)

	return s

def build_vocab(words):
	word_list = []

	stop_words = set(stopwords.words('english')) 
	for sent in words:
		for word in sent.lower().split():
			word = preprocess_string(word)
			if word not in stop_words and word != '':
				word_list.append(word)

	corpus = Counter(word_list)
	# sorting on the basis of most common words
	corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:2000]
	# creating a dict
	vocab = {w:i+1 for i,w in enumerate(corpus_)}
	return vocab

def sequence_pad(sentences, seq_len):
	"""Pads sequences of tokenized sentences to a fixed length.

	Args:
	sentences: List of lists, where each inner list contains integer 
		tokens representing a tokenized sentence.
	seq_len: Maximum length to pad/truncate sentences to.
	
	Returns:
	features: 2D numpy array with shape (num_sentences, seq_len) 
		containing the padded sentence tokens."""

	features = np.zeros((len(sentences), seq_len),dtype=int)
	for ii, review in enumerate(sentences):
		if len(review) != 0:
			features[ii, -len(review):] = np.array(review)[:seq_len]
	return features



def tokenize(pre):
	# 
	vocab = build_vocab(x_train)
	final_list_train,final_list_test = [],[]
	for sent in x_train:
			final_list_train.append([vocab[preprocess_string(word)] for word in sent.lower().split() 
									 if preprocess_string(word) in vocab.keys()])
	for sent in x_val:
			final_list_test.append([vocab[preprocess_string(word)] for word in sent.lower().split() 
									if preprocess_string(word) in vocab.keys()])
			
	encoded_train = [1 if label =='positive' else 0 for label in y_train]  
	encoded_test = [1 if label =='positive' else 0 for label in y_val] 
	print("Number of words in vocabulary:", len(vocab))
	print(np.array(len(encoded_train)))
	print(np.array(len(encoded_test)))
	print(np.array(len(final_list_train)))
	print(np.array(len(final_list_test)))
	return final_list_train, encoded_train,final_list_test, encoded_test



class SentimentRNN(nn.Module):
	def __init__(self,no_layers,vocab_size,output_dim, hidden_dim,embedding_dim,drop_prob=0.5):
		super(SentimentRNN,self).__init__()
		self.device = torch.device('cpu')
		self.output_dim = output_dim
		self.hidden_dim = hidden_dim

		self.no_layers = no_layers
		self.vocab_size = vocab_size
	
		# embedding and LSTM layers
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		
		#lstm
		self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=self.hidden_dim,
						num_layers=no_layers, batch_first=True)
		
		
		# dropout layer
		self.dropout = nn.Dropout(0.3)
	
		# linear and sigmoid layer
		self.fc = nn.Linear(self.hidden_dim, output_dim)
		self.sig = nn.Sigmoid()
		
	def forward(self,x,hidden):
		batch_size = x.size(0)
		# embeddings and lstm_out
		embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True
		#print(embeds.shape)  #[50, 500, 1000]
		lstm_out, hidden = self.lstm(embeds, hidden)
		
		lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) 
		
		# dropout and fully connected layer
		out = self.dropout(lstm_out)
		out = self.fc(out)
		
		# sigmoid function
		sig_out = self.sig(out)
		
		# reshape to be batch_size first
		sig_out = sig_out.view(batch_size, -1)

		sig_out = sig_out[:, -1] # get last batch of labels
		
		# return last sigmoid output and hidden state
		return sig_out, hidden
		
		
		
	def init_hidden(self, batch_size):
		''' Initializes hidden state '''
		# Create two new tensors with sizes n_layers x batch_size x hidden_dim,
		# initialized to zero, for hidden state and cell state of LSTM
		h0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(self.device)
		c0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(self.device)
		hidden = (h0,c0)
		return hidden

def load_model(vocab, path):

	no_layers = 2
	vocab_size = len(vocab) + 1 #extra 1 for padding
	embedding_dim = 64
	output_dim = 1
	hidden_dim = 256

	model = SentimentRNN(no_layers=no_layers,vocab_size=vocab_size,output_dim=output_dim,hidden_dim=hidden_dim,embedding_dim=embedding_dim,drop_prob=0.5)
	model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
	model.cpu()
	model.eval()
	return model
	

def pred(model,vocab, text):
	"""Predicts sentiment of input text using the provided RNN model, vocabulary, and text input"""
	device = torch.device('cpu')
	word_seq = np.array([vocab[preprocess_string(word)] for word in text.split() 
						if preprocess_string(word) in vocab.keys()])
	word_seq = np.expand_dims(word_seq,axis=0)
	pad =  torch.from_numpy(sequence_pad(word_seq,500))
	inputs = pad.to(device)
	batch_size = 1	
	h = model.init_hidden(batch_size)
	h = tuple([each.data for each in h])
	output, h = model(inputs, h)
	pro = output.item()
	sentiment = "positive" if pro > 0.5 else "negative"
	pro = (1 - pro) if sentiment == "negative" else pro
	return pro,sentiment

