import math
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as pad
from layers import *
import datalib.constants as constants
from train_utils import check_values


class cnn_classifier(nn.Module):
	"""docstring for cnn_classifier"""
	def __init__(self, vocab_size, emb_size, emb_max_norm, filter_sizes, n_filters, leaky, pad, dropout_rate, output_size):
		super(cnn_classifier, self).__init__()
		
		self.emb = nn.Embedding(vocab_size, emb_size, constants.PAD_ID, max_norm = emb_max_norm)
		self.cnn = cnn(emb_size, filter_sizes, n_filters, leaky, pad, dropout_rate, output_size)
	def forward(self, inputs, soft_input=False):
		x, padding_mask = inputs['x_b'], inputs['padding_mask_b']
		# if not batch_first:
		# 	x = x.transpose(0, 1).contiguous()
		# 	if padding_mask is not None:
		# 		padding_mask = padding_mask.transpose(0, 1).contiguous()

		if soft_input:
			x = torch.matmul(x, self.emb.weight)
		else:
			x = self.emb(x)

		logits = self.cnn(x, padding_mask)
		return logits
		
class attn_classifier(nn.Module):
	"""docstring for attn_classifier"""
	def __init__(self, vocab_size, emb_size, emb_max_norm, 
		hid_size, rnn_type, bidirectional, bilin_att, self_att, dropout_rate, output_size):
		super(attn_classifier, self).__init__()
		self.hid_size = hid_size
		self.rnn_type = rnn_type
		self.bidirectional = bidirectional
		
		self.emb = nn.Embedding(vocab_size, emb_size, constants.PAD_ID, max_norm = emb_max_norm)
		
		self.encoder = getattr(nn, rnn_type)(emb_size, hid_size, bidirectional = bidirectional)
		self.self_att = self_att
		if self_att:
			assert not bilin_att, 'bilin_att does not support self attention mode!'
		rnn_size = hid_size*2 if bidirectional else hid_size
		self.attention = bilinear_attention(rnn_size, rnn_size) if bilin_att else feedforward_attention(rnn_size, rnn_size, rnn_size, self_att)
		self.dropout = nn.Dropout(dropout_rate)
		self.projection = nn.Linear(rnn_size, output_size)

	def reshape_final_state(self, final_state):
		if self.rnn_type == 'LSTM':
			final_state = final_state[0]
		if self.bidirectional:
			return final_state.transpose(0, 1).contiguous().view(-1, 2*self.hid_size)
		else:
			return final_state.squeeze(0)
	
	def forward(self, inputs, soft_input = False):
		x, enc_padding_mask = inputs['x'], inputs['padding_mask']
		# if batch_first:
		# 	x = x.transpose(0, 1).contiguous()
		# 	if enc_padding_mask is not None:
		# 		enc_padding_mask = enc_padding_mask.transpose(0, 1).contiguous()
		if soft_input:
			x = torch.matmul(x, self.emb.weight)
		else:
			x = self.emb(x)
			
		outputs, final_state = self.encoder(pack(x, enc_padding_mask.bitwise_not().long().sum(0), enforce_sorted=False))
		outputs = pad(outputs, total_length=x.size(0))[0]
		# check_values(outputs, 'ac outputs', False)
		# check_values(final_state, 'ac final_state', False)
		attn_dist = self.attention(outputs, enc_padding_mask, None if self.self_att else self.reshape_final_state(final_state))
		# check_values(attn_dist, 'ac attn_dist', False)
		# if att_only:
		# 	return attn_dist

		avg_state = torch.sum(attn_dist.unsqueeze(-1) * outputs, 0)
		avg_state = self.dropout(avg_state)
		logits = self.projection(avg_state)

		return logits#, attn_dist

class Transformer_classifier(nn.Module):
	"""docstring for Transformer_classifier"""
	def __init__(self, vocab_size, emb_size, emb_max_norm, pos_sincode, token_emb_scale, pos_max_len, num_heads, hid_size, num_layers, 
		dropout_rate, output_size, with_cls, att_dropout_rate, transformer_norm_bf):
		super(Transformer_classifier, self).__init__()
		self.emb_size = emb_size
		self.emb = nn.Embedding(vocab_size, emb_size, constants.PAD_ID, max_norm = emb_max_norm)
		# self.pos_emb = PositionalEncoding(emb_size, dropout_rate, pos_max_len) if pos_sincode else PositionalEmbedding(emb_size, emb_max_norm, dropout_rate, pos_max_len)
		self.pos_emb = PositionalEmbedding(pos_max_len, emb_size, emb_max_norm, not pos_sincode)
		self.token_emb_scale = token_emb_scale
		# encoder_layer = nn.TransformerEncoderLayer(emb_size, num_heads, hid_size, dropout_rate)
		# self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
		self.encoder = TransformerEncoder(emb_size, hid_size, num_heads, att_dropout_rate, dropout_rate, transformer_norm_bf, num_layers)
		self.with_cls = with_cls
		# if with_cls:
		# 	self.cls_tok = None
		# 	self.cls_mask = None
		self.projection = nn.Linear(emb_size, output_size)
		self.dropout_rate = dropout_rate


	def forward(self, inputs, soft_input = False):
		x, enc_padding_mask, lens = inputs['x'], inputs['padding_mask_b'], inputs['lens']
		# if batch_first:
		# 	x = x.transpose(0, 1).contiguous()
		# 	if enc_padding_mask is not None:
		# 		enc_padding_mask = enc_padding_mask.transpose(0, 1).contiguous()
		if not self.with_cls:
			all_zero_mask = lens == 0
			if all_zero_mask.any():
				enc_padding_mask = enc_padding_mask.masked_fill(all_zero_mask.unsqueeze(-1), False)
		if self.with_cls:
			bsz = x.size(1)
			# if self.cls_tok is None or self.cls_tok.size(1) != bsz:
			if soft_input:
				cls_tok = x.new_zeros((1, bsz, x.size(2)))
				cls_tok[:, :, constants.BOS_ID] = 1
			else:
				cls_tok = x.new_full((1, bsz), constants.BOS_ID)
			cls_mask = enc_padding_mask.new_zeros((bsz, 1))
			x = torch.cat([cls_tok, x], 0)
			enc_padding_mask = torch.cat([cls_mask, enc_padding_mask], 1)

		if soft_input:
			x = torch.matmul(x, self.emb.weight)
		else:
			x = self.emb(x)
		if self.token_emb_scale:
			x = x * math.sqrt(self.emb_size)
		x = x + self.pos_emb(x.size(0), device=x.device)
		x = F.dropout(x, p=self.dropout_rate, training=self.training)
		enc_outputs = self.encoder(x, encoder_padding_mask=enc_padding_mask)
		if self.with_cls:
			state = enc_outputs[0]
		else:
			enc_outputs = enc_outputs.masked_fill(enc_padding_mask.t().unsqueeze(-1), 0)
			state = enc_outputs.sum(0) / lens.float().masked_fill(all_zero_mask, 1).unsqueeze(-1)

		logits = self.projection(state)
		return logits



		


def add_one_class(model):
	assert isinstance(model, attn_classifier) or isinstance(model, cnn_classifier) or isinstance(model, Transformer_classifier)
	if isinstance(model, attn_classifier) or isinstance(model, Transformer_classifier):
		device = model.projection.weight.device
		output_size, input_size = model.projection.weight.size()
		output_size = output_size + 1
		model.projection = nn.Linear(input_size, output_size).to(device)
	else:
		device = model.cnn.linear.weight.device
		output_size, input_size = model.cnn.linear.weight.size()
		output_size = output_size + 1
		model.cnn.linear = nn.Linear(input_size, output_size).to(device)

def change_output_size(model, n_targets):
	assert isinstance(model, attn_classifier) or isinstance(model, cnn_classifier) or isinstance(model, Transformer_classifier)
	if isinstance(model, attn_classifier) or isinstance(model, Transformer_classifier):
		device = model.projection.weight.device
		output_size, input_size = model.projection.weight.size()
		output_size = n_targets
		model.projection = nn.Linear(input_size, output_size).to(device)
	else:
		device = model.cnn.linear.weight.device
		output_size, input_size = model.cnn.linear.weight.size()
		output_size = n_targets
		model.cnn.linear = nn.Linear(input_size, output_size).to(device)