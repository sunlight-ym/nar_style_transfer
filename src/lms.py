import math
import torch
import torch.nn as nn
# import torch.nn.functional as F
from layers import *
# import datalib.constants as constants
from train_utils import check_values


class rnn_lm(nn.Module):
	"""docstring for rnn_lm"""
	def __init__(self, vocab_size, emb_size, emb_max_norm, hid_size, rnn_type, dropout_rate, num_bias, tie_weights):
		super(rnn_lm, self).__init__()
		# self.vocab_size = vocab_size
		# self.emb_size = emb_size
		# self.hid_size = hid_size
		# self.rnn_type = rnn_type
		self.emb = nn.Embedding(vocab_size, emb_size, constants.PAD_ID, max_norm = emb_max_norm)
		self.encoder = getattr(nn, rnn_type)(emb_size, hid_size)
		self.dropout = nn.Dropout(dropout_rate)
		# self.projection = nn.Linear(hid_size, vocab_size)
		self.projection = multi_bias_linear(num_bias, hid_size, vocab_size)
		if tie_weights:
			assert emb_size == hid_size, 'tied weights require emb_size == hid_size'
			self.projection.linear.weight = self.emb.weight
	
	def forward(self, inputs, soft_input=False):
		x, inds = inputs['x'], inputs['inds']
		# if batch_first:
		# 	x = x.transpose(0, 1).contiguous()
		if soft_input:
			x = torch.matmul(x, self.emb.weight)
		else:
			x = self.emb(x)
		outputs, _ = self.encoder(x)
		outputs = self.dropout(outputs)
		logits = self.projection(outputs, inds)

		return logits

class Transformer_lm(nn.Module):
	"""docstring for Transformer_lm"""
	def __init__(self, vocab_size, emb_size, emb_max_norm, pos_sincode, token_emb_scale, pos_max_len, num_heads, hid_size, num_layers, dropout_rate, num_bias, tie_weights,
		att_dropout_rate, transformer_norm_bf):
		super(Transformer_lm, self).__init__()
		self.emb_size = emb_size
		self.emb = nn.Embedding(vocab_size, emb_size, constants.PAD_ID, max_norm = emb_max_norm)
		# self.pos_emb = PositionalEncoding(emb_size, dropout_rate, pos_max_len) if pos_sincode else PositionalEmbedding(emb_size, emb_max_norm, dropout_rate, pos_max_len)
		self.pos_emb = PositionalEmbedding(pos_max_len, emb_size, emb_max_norm, not pos_sincode)
		self.token_emb_scale = token_emb_scale
		# encoder_layer = nn.TransformerEncoderLayer(emb_size, num_heads, hid_size, dropout_rate)
		# self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
		self.encoder = TransformerEncoder(emb_size, hid_size, num_heads, att_dropout_rate, dropout_rate, transformer_norm_bf, num_layers)
		self.projection = multi_bias_linear(num_bias, emb_size, vocab_size)
		self.dropout_rate = dropout_rate
		if tie_weights:
			self.projection.linear.weight = self.emb.weight
		# self.seq_mask = None
	def forward(self, inputs, soft_input=False):
		x, inds = inputs['x'], inputs['inds']
		# if batch_first:
		# 	x = x.transpose(0, 1).contiguous()
		if soft_input:
			x = torch.matmul(x, self.emb.weight)
		else:
			x = self.emb(x)
		if self.token_emb_scale:
			x = x * math.sqrt(self.emb_size)
		x = x + self.pos_emb(x.size(0), device=x.device)
		x = F.dropout(x, p=self.dropout_rate, training=self.training)
		
		seq_mask = generate_square_subsequent_mask(x.size(0), x.device)
		enc_outputs = self.encoder(x, seq_mask)
		logits = self.projection(enc_outputs, inds)
		return logits