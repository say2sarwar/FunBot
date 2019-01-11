import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F 
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math





class Attn(torch.nn.Module):
	"""docstring for Attn"""
	def __init__(self, method, hidden_size):
		super(Attn, self).__init__()
		self.method = method
		if self.method not in ['dot', 'general', 'concat']:
			raise ValueError(self.method, 'is not an appropriate attention method')
		self.hidden_size = hidden_size
		if self.method == 'general':
			self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
		elif self.method == 'concat':
			self.attn = torch.nn.Linear(self.hidden_size*2, hidden_size)
			self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

	def dot_score(self, hidden, encoder_output):
		return torch.sum(hidden*encoder_output, dim=2)

	def general_score(self, hidden, encoder_output):
		energy = self.attn(encoder_output)
		return torch.sum(hidden*energy, dim=2)

	def concat_score(self, hidden, encoder_output):
		energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1),
			encoder_output), 2)).tanh()

		return torch.sum(self.v*energy, dim=2)


	def forward(self, hidden, encoder_output):

		if self.method == 'general':
			attn_energies = self.general_score(hidden, encoder_output)
		elif self.method == 'concat':
			attn_energies = self.concat_score(hidden, encoder_output)
		elif self.method == 'dot':
			attn_energies = self.dot_score(hidden, encoder_output)

		attn_energies = attn_energies.t()

		return F.softmax(attn_energies, dim=1).unsqueeze(1)


class RNNEncoder(nn.Module):
	"""docstring for RNNEncoder"""
	def __init__(self, hidden_size, embedding, n_layers=1,dropout=0):
		super(RNNEncoder, self).__init__()
		self.n_layers = n_layers
		self.hidden_size = hidden_size
		self.embedding = embedding

		self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
			dropout=(0 if n_layers ==1 else dropout), bidirectional=True)

	def forward(self, input_seq, input_lengths, hidden=None):
		embedded = self.embedding(input_seq)

		packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

		outputs, hidden = self.gru(packed, hidden)

		outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)

		outputs = outputs[:, :, :self.hidden_size] + outputs[:,: ,self.hidden_size:]

		return outputs, hidden


class RnnDecoder(nn.Module):
	"""docstring for RnnDecoder"""
	def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
		super(RnnDecoder, self).__init__()
		self.attn_model = attn_model
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.dropout = dropout

		self.embedding = embedding
		self.embedding_dropout = nn.Dropout(dropout)
		self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1
			else dropout))
		self.concat = nn.Linear(hidden_size*2, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		self.attn = Attn(attn_model, hidden_size)

	def forward(self, input_step, last_hidden, encoder_outputs):

		embedded = self.embedding(input_step)
		embedded = self.embedding_dropout(embedded)

		rnn_output, hidden = self.gru(embedded, last_hidden)

		attn_weights = self.attn(rnn_output, encoder_outputs)
		context = attn_weights.bmm(encoder_outputs.transpose(0,1))

		rnn_output = rnn_output.squeeze(0)
		context = context.squeeze(1)
		concat_input = torch.cat((rnn_output, context), 1)
		concat_output = torch.tanh(self.concat(concat_input))

		output = self.out(concat_output)
		output = F.softmax(output, dim=1)

		return output, hidden 




		
		
		