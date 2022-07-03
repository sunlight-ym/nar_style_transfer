import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import datalib.constants as constants
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as pad
# import spacy
# from spacy.tokens import Doc

def acc_steps(result, step_result):
	for k in result:
		result[k].append(step_result[k])
def acc_steps_tensor(result, step_result, ind):
	for k in result:
		result[k][ind] = step_result[k]
def steps_to_tensor(result):
	for k in result:
		result[k] = torch.stack(result[k])
def init_dict(result, key, cond=True):
	if cond:
		result[key] = []
def _repeat(num, batch_dim, x):
	if x is None:
		return x
	shape = list(x.size())
	x = x.unsqueeze(batch_dim+1)
	new_shape = list(x.size())
	new_shape[batch_dim+1] = num
	x = x.expand(*new_shape).contiguous()
	shape[batch_dim] = shape[batch_dim]*num
	return x.view(shape)

def repeat_tensors(num, batch_dim, mats):
	if num == 1 or (mats is None):
		return mats
	elif torch.is_tensor(mats):
		return _repeat(num, batch_dim, mats)
	else:
		return [_repeat(num, batch_dim, m) for m in mats]



def softmax_sample(logits, tau = 1, gumbel = True, dim=-1):
	tau = max(tau, 1e-10)
	if gumbel:
		return F.gumbel_softmax(logits, tau=tau, dim=dim)
	else:
		return F.softmax(logits / tau, dim)


def init_rnn_hidden(batch_size, n_layers, n_directions, hid_size, rnn_type, input_tensor):
	size=(n_layers*n_directions, batch_size, hid_size)
	if rnn_type == 'LSTM':
		return input_tensor.new_zeros(size), input_tensor.new_zeros(size)
	else:
		return input_tensor.new_zeros(size)

def cat_zeros_at_start(m):
	bsz = m.size(0)
	start = m.new_zeros((bsz, 1))
	return torch.cat([start, m], 1)



def get_out_lens(x, seq_dim=0, exceed_mask=None, return_with_eos=True):
	eos_mask = (x == constants.EOS_ID)
	if exceed_mask is not None:
		eos_mask = eos_mask | exceed_mask
	eos_mask = eos_mask.int()
	mask_sum = eos_mask.cumsum(seq_dim)
	# eos_mask = eos_mask.masked_fill_(mask_sum != 1, 0)
	# lens = eos_mask.argmax(seq_dim)
	lens = torch.sum((mask_sum == 0).long(), seq_dim)
	# if count_eos:
	if not return_with_eos:
		return lens
	lens_with_eos = lens + 1
	lens_with_eos.clamp_(max=x.size(seq_dim))
	# zl_mask = eos_mask.sum(seq_dim) == 0
	# lens[zl_mask] = x.size(seq_dim)
	return lens, lens_with_eos




def get_pad_size(mode, k_size):
	if mode is None:
		return 0
	elif mode=='half':
		return (int((k_size-1)/2), 0)
	elif mode=='full':
		return (k_size-1, 0)
	else:
		raise ValueError('unsupported padding mode')

def get_padding_mask(x, lens, batch_dim=1, seq_dim=0):
	max_len = x.size(seq_dim)
	mask = torch.arange(0, max_len, dtype = torch.long, device = x.device).unsqueeze(batch_dim)#.expand(max_len, lens.size(0))
	mask = mask >= lens.unsqueeze(seq_dim)
	return mask

def get_padding_mask_on_size(max_len, lens, batch_dim=1, seq_dim=0, flip=False):
	# max_len = x.size(seq_dim)
	mask = torch.arange(0, max_len, dtype = torch.long, device = lens.device).unsqueeze(batch_dim)#.expand(max_len, lens.size(0))
	if flip:
		mask = mask < lens.unsqueeze(seq_dim)
	else:
		mask = mask >= lens.unsqueeze(seq_dim)
	return mask

def get_padding_masks(x, lens, lens_with_eos, batch_dim=1, seq_dim=0):
	max_len = x.size(seq_dim)
	inds = torch.arange(0, max_len, dtype = torch.long, device = x.device).unsqueeze(batch_dim)#.expand(max_len, lens.size(0))
	mask = inds >= lens.unsqueeze(seq_dim)
	mask_with_eos = inds >= lens_with_eos.unsqueeze(seq_dim)
	return mask, mask_with_eos

def reverse_seq(x, lens, eos_mask=None, batch_dim=1, seq_dim=0):
	#lens should be without eos
	inds = lens.unsqueeze(seq_dim)
	inds = inds - torch.arange(1, x.size(seq_dim)+1, dtype = torch.long, device = x.device).unsqueeze(batch_dim)
	inds.masked_fill_(inds < 0, 0)
	rev_x = x.gather(seq_dim, inds)
	if eos_mask is not None:
		rev_x = rev_x.masked_fill(eos_mask, constants.EOS_ID)
	return rev_x

		
def conv_mask(x, conv_pad_num, padding_mask):
	if conv_pad_num > 0:
		prefix = padding_mask.new_full((x.size(0), conv_pad_num), False)
		padding_mask = torch.cat([prefix, padding_mask], 1)
	mask = padding_mask[:,:x.size(2)]
	mask = mask.unsqueeze(1)
	low_value = x.detach().min() - 1
	# print('x', x.size())
	# print('low_values', low_values.size())
	# print('mask', mask.size())
	# low_values = low_values.expand_as(x).masked_select(mask)
	return x.masked_fill(mask, low_value)

class cnn(nn.Module):
	
	def __init__(self, input_size, filter_sizes, n_filters, leaky, pad, dropout_rate, output_size):
		super(cnn, self).__init__()

		self.conv_pad_nums = [w-1 if pad else 0 for w in filter_sizes]

		self.convs = nn.ModuleList([
			nn.Conv2d(1, n_filters, (w, input_size), padding = (self.conv_pad_nums[i], 0)) 
			for i, w in enumerate(filter_sizes)])
		self.leaky = leaky
		self.dropout = nn.Dropout(dropout_rate)
		self.linear = nn.Linear(len(filter_sizes)*n_filters, output_size)
	
	def forward(self, x, padding_mask):
		# x is the embedding matrix: seq * batch * emb_size
		
		x = x.unsqueeze(1)
		conv_outs=[]
		for i, conv in enumerate(self.convs):
			conv_out = conv(x)
			conv_out = F.leaky_relu(conv_out) if self.leaky else F.relu(conv_out)
			conv_out = conv_out.squeeze(-1)
			if padding_mask is not None:
				conv_out = conv_mask(conv_out, self.conv_pad_nums[i], padding_mask)
			conv_out = conv_out.max(2)[0]
			conv_outs.append(conv_out)
		conv_outs = torch.cat(conv_outs, 1)
		conv_outs = self.dropout(conv_outs)
		logits = self.linear(conv_outs)

		return logits

class feedforward_attention(nn.Module):
	"""docstring for feedforward_attention"""
	def __init__(self, input_size, hid_size, att_size, self_att = False, use_coverage = False, eps = 1e-10):
		super(feedforward_attention, self).__init__()
		self.input_proj = nn.Linear(input_size, att_size, bias = self_att)
		self.self_att = self_att
		if not self_att:
			self.hid_proj = nn.Linear(hid_size, att_size)
		self.use_coverage = use_coverage
		if use_coverage:
			self.cov_proj = nn.Linear(1, att_size)
		self.v = nn.Linear(att_size, 1, bias = False)
		self.eps = eps
	
	def forward(self, enc_outputs, enc_padding_mask, hidden = None, coverage = None):
		att_fea = self.input_proj(enc_outputs) # seq * b * att_size
		if not self.self_att:
			hid_fea = self.hid_proj(hidden) # b * att_size
			att_fea = att_fea + hid_fea.unsqueeze(0)
		if self.use_coverage:
			cov_fea = self.cov_proj(coverage.unsqueeze(-1))
			att_fea = att_fea + cov_fea

		scores = self.v(torch.tanh(att_fea)).squeeze(-1) # seq * b

		attn_dist = F.softmax(scores, dim=0)
		if enc_padding_mask is not None:
			attn_dist = attn_dist.masked_fill(enc_padding_mask, 0)
			normalization_factor = attn_dist.sum(0, keepdim = True)
			attn_dist = attn_dist / (normalization_factor + self.eps)

		return attn_dist

class bilinear_attention(nn.Module):
	"""docstring for bilinear_attention"""
	def __init__(self, input_size, hid_size, eps = 1e-10):
		super(bilinear_attention, self).__init__()
		self.proj = nn.Linear(hid_size, input_size, bias=False)
		self.eps = eps
	
	def forward(self, enc_outputs, enc_padding_mask, hidden, dummy = None):
		hid_fea = self.proj(hidden).unsqueeze(0)
		scores = torch.sum(hid_fea * enc_outputs, -1)

		attn_dist = F.softmax(scores, dim=0)
		if enc_padding_mask is not None:
			attn_dist = attn_dist.masked_fill(enc_padding_mask, 0)
			normalization_factor = attn_dist.sum(0, keepdim = True)
			attn_dist = attn_dist / (normalization_factor + self.eps)
		
		return attn_dist

class attn_decoder(nn.Module):
	"""docstring for attn_decoder"""
	def __init__(self, emb_size, hid_size, num_layers, rnn_type, bilin_att, enc_size, feed_last_context, att_coverage = False, eps = 1e-10, use_att=True, use_copy=False, cxt_drop=0):
		super(attn_decoder, self).__init__()
		self.rnn_type = rnn_type
		self.use_att = use_att
		self.use_copy = use_copy
		self.feed_last_context = feed_last_context
		self.cxt_drop = cxt_drop
		if att_coverage:
			assert not bilin_att, 'bilin_att does not support using coverage!'
		self.rnn = getattr(nn, rnn_type)(emb_size+enc_size if feed_last_context and self.use_att else emb_size, hid_size, num_layers = num_layers)
		if use_att:
			self.attention = bilinear_attention(enc_size, hid_size, eps=eps) if bilin_att else feedforward_attention(enc_size, hid_size, hid_size, use_coverage=att_coverage, eps=eps)
			if use_copy:
				self.copy = nn.Linear((enc_size * 2 if feed_last_context else enc_size) + hid_size + emb_size, 1)
		# self.out1 = nn.Linear(enc_size + hid_size, hid_size)
		# self.out2 = nn.Linear(hid_size, vocab_size)

	def forward(self, input_emb, last_state, enc_outputs, enc_padding_mask, last_context, coverage = None):
		if self.use_att and self.feed_last_context:
			input_emb = torch.cat([last_context, input_emb], 1)
		self.rnn.flatten_parameters()
		dec_out, state = self.rnn(input_emb.unsqueeze(0), last_state)
		if self.use_att:
			att_hid = state[0][-1] if self.rnn_type == 'LSTM' else state[-1]
			attn_dist = self.attention(enc_outputs, enc_padding_mask, att_hid, coverage)
			context = torch.sum(attn_dist.unsqueeze(-1) * enc_outputs, 0)
			if self.cxt_drop > 0:
				context = F.dropout(context, self.cxt_drop, training=self.training)

			if self.use_copy:
				p_copy_input = torch.cat([context, att_hid, input_emb], 1)
				p_copy = self.copy(p_copy_input)
				p_copy = torch.sigmoid(p_copy.squeeze(-1))
		
		if not self.use_att:
			attn_dist, context, p_copy = None, None, None
		if self.use_att and not self.use_copy:
			p_copy = None

		# pred_input = torch.cat([context, att_hid], 1)
		# pred = self.out2(self.out1(pred_input))
		# vocab_dist = F.softmax(pred, dim = 1)

		return state, attn_dist, context, p_copy

class multi_bias_linear(nn.Module):
	"""docstring for multi_bias_linear"""
	def __init__(self, num_bias, input_size, output_size):
		super(multi_bias_linear, self).__init__()
		self.num_bias = num_bias
		self.input_size = input_size
		self.output_size = output_size
		if num_bias == 1:
			self.linear = Linear(input_size, output_size)
		else:
			self.linear = Linear(input_size, output_size, bias = False)
			self.biases = nn.Parameter(torch.Tensor(num_bias, output_size))
			self.init_bias()
	def init_bias(self):
		# fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.linear.weight)
		# bound = 1 / math.sqrt(fan_in)
		# nn.init.uniform_(self.biases, -bound, bound)
		nn.init.constant_(self.biases, 0.)

	def forward(self, x, inds=None, soft_inds=False):
		x = self.linear(x)
		if self.num_bias == 1:
			return x
		elif soft_inds:
			return x + torch.mm(inds, self.biases)
		else:
			return x + self.biases[inds]



def generate_square_subsequent_mask(sz, device):
	r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
		Unmasked positions are filled with float(0.0).
	"""
	mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
	mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
	return mask

def generate_self_mask(sz, device, escape_start):
	# dim = tensor.size(0)
	eye_matrix = torch.eye(sz, dtype=torch.float, device=device)
	eye_matrix[eye_matrix == 1.0] = float('-inf')
	if escape_start:
		eye_matrix[0, 0] = 0
	return eye_matrix


# def get_clones(module, N):
# 	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class LearnedPositionalEmbedding(nn.Embedding):

	def __init__(self, num_embeddings, embedding_dim, max_norm):

		super().__init__(num_embeddings+1, embedding_dim, 0, max_norm)
		nn.init.normal_(self.weight, mean=0, std=embedding_dim ** -0.5)
		nn.init.constant_(self.weight[0], 0)

	def forward(self, seq_len=None, j=None, x_mask=None, device=None, detail_pos=None):
		"""Input is expected to be of size [seqlen x bsz] or [bsz] with j."""
		if detail_pos is not None:
			positions = detail_pos + 1
		elif j is None:
			# assert len(x_size) == 2
			positions = torch.arange(1, seq_len+1, dtype=torch.long, device=device).unsqueeze(1) # seq, 1
			if x_mask is not None:
				positions = positions.masked_fill(x_mask, 0) # seq, bsz
		else:
			# assert len(x_size) == 1
			positions = torch.tensor([j+1], dtype=torch.long, device=device) # 1

		return super().forward(positions)

class SinusoidalPositionalEmbedding(nn.Module):

	def __init__(self, num_embeddings, embedding_dim):
		super().__init__()
		weights = SinusoidalPositionalEmbedding.get_embedding(num_embeddings, embedding_dim)
		self.register_buffer('weights', weights)
	
	@staticmethod
	def get_embedding(num_embeddings, embedding_dim):
		
		assert embedding_dim % 2 == 0
		half_dim = embedding_dim // 2
		emb = math.log(10000) / (half_dim - 1)
		emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
		emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
		emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
		return emb

	def forward(self, seq_len=None, j=None, x_mask=None, device=None, detail_pos=None):
		if detail_pos is not None:
			emb = self.weights[detail_pos.view(-1)].view(*detail_pos.size(), -1)
		elif j is None:
			# assert len(x_size) == 2
			emb = self.weights[:seq_len].unsqueeze(1) # seq, 1, d
			if x_mask is not None:
				emb = emb.masked_fill(x_mask.unsqueeze(-1), 0) # seq, bsz, d

		else:
			# assert len(x_size) == 1
			emb = self.weights[j].unsqueeze(0) # 1, d

		return emb

def PositionalEmbedding(num_embeddings, embedding_dim, max_norm, learned):
	num_embeddings = 256 if num_embeddings is None else num_embeddings
	if learned:
		return LearnedPositionalEmbedding(num_embeddings, embedding_dim, max_norm)
	else:
		return SinusoidalPositionalEmbedding(num_embeddings, embedding_dim)

def Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None):
	m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm)
	nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
	nn.init.constant_(m.weight[padding_idx], 0)
	return m

def Linear(in_features, out_features, bias=True):
	m = nn.Linear(in_features, out_features, bias)
	nn.init.xavier_uniform_(m.weight)
	if bias:
		nn.init.constant_(m.bias, 0.)
	return m

class TransformerEncoderLayer(nn.Module):
	

	def __init__(self, encoder_embed_dim, encoder_ffn_embed_dim, encoder_attention_heads, attention_dropout, dropout, encoder_normalize_before):
		super().__init__()
		self.embed_dim = encoder_embed_dim
		self.self_attn = nn.MultiheadAttention(
			self.embed_dim, encoder_attention_heads,
			dropout=attention_dropout)
		self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
		self.dropout = dropout
		self.activation_fn = F.relu
		
		self.normalize_before = encoder_normalize_before
		self.fc1 = Linear(self.embed_dim, encoder_ffn_embed_dim)
		self.fc2 = Linear(encoder_ffn_embed_dim, self.embed_dim)
		self.final_layer_norm = nn.LayerNorm(self.embed_dim)

	def forward(self, x, src_mask, encoder_padding_mask):
		"""
		Args:
			x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
			encoder_padding_mask (ByteTensor): binary ByteTensor of shape
				`(batch, src_len)` where padding elements are indicated by ``1``.
		Returns:
			encoded output of shape `(batch, src_len, embed_dim)`
		"""
		residual = x
		x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
		x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=src_mask, key_padding_mask=encoder_padding_mask)
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = residual + x
		x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

		residual = x
		x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
		x = self.activation_fn(self.fc1(x))
		x = self.fc2(x)
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = residual + x
		x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
		return x

	def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
		assert before ^ after
		if after ^ self.normalize_before:
			return layer_norm(x)
		else:
			return x
	
class TransformerDecoderLayer(nn.Module):

	def __init__(self, decoder_embed_dim, decoder_ffn_embed_dim, decoder_attention_heads, attention_dropout, dropout, decoder_normalize_before, positional_attention):
		super().__init__()
		self.embed_dim = decoder_embed_dim
		self.self_attn = nn.MultiheadAttention(
			self.embed_dim, decoder_attention_heads,
			dropout=attention_dropout)
		self.dropout = dropout
		self.activation_fn = F.relu
		self.normalize_before = decoder_normalize_before

		self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

		self.positional_attention = positional_attention
		if self.positional_attention:
			self.position_attn = nn.MultiheadAttention(
				self.embed_dim, decoder_attention_heads,
				dropout=attention_dropout)
			self.position_layer_norm = nn.LayerNorm(self.embed_dim)
		
		self.encoder_attn = nn.MultiheadAttention(
			self.embed_dim, decoder_attention_heads,
			dropout=attention_dropout)
		self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

		self.fc1 = Linear(self.embed_dim, decoder_ffn_embed_dim)
		self.fc2 = Linear(decoder_ffn_embed_dim, self.embed_dim)

		self.final_layer_norm = nn.LayerNorm(self.embed_dim)

	def forward(
		self,
		x,
		encoder_out,
		past_self_in=None,
		past_pos_in=None,
		ret_past=False,
		encoder_padding_mask=None,
		self_attn_mask=None,
		position_embedding=None,
		targets_padding=None,
		need_self_attn_weights=False,
		need_pos_attn_weights=False,
		need_enc_attn_weights=False
	):
		"""
		Args:
			x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
			encoder_padding_mask (ByteTensor): binary ByteTensor of shape
				`(batch, src_len)` where padding elements are indicated by ``1``.
		Returns:
			encoded output of shape `(batch, src_len, embed_dim)`
		"""
		attn_weights, past_states = {}, {}
		residual = x
		x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
		self_kv = x if past_self_in is None else torch.cat([past_self_in, x], 0)
		if ret_past:
			past_states['self_in'] = self_kv
		x, attn_weights['self_attn_weights'] = self.self_attn(
			query=x,
			key=self_kv,
			value=self_kv,
			key_padding_mask=targets_padding,
			need_weights=need_self_attn_weights,
			attn_mask=self_attn_mask,
		)
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = residual + x
		x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

		if self.positional_attention:
			if position_embedding is None:
				raise ValueError('position embedding is not provided')
			residual = x
			x = self.maybe_layer_norm(self.position_layer_norm, x, before = True)
			pos_v = x if past_pos_in is None else torch.cat([past_pos_in, x], 0)
			if ret_past:
				past_states['pos_in'] = pos_v
			x, attn_weights['pos_attn_weights'] = self.position_attn(
				query=position_embedding if past_pos_in is None else position_embedding[-1:],
				key=position_embedding,
				value=pos_v,
				key_padding_mask=targets_padding,
				need_weights=need_pos_attn_weights,
				attn_mask=self_attn_mask,
			)
			x = F.dropout(x, p=self.dropout, training=self.training)
			x = residual + x
			x = self.maybe_layer_norm(self.position_layer_norm, x, after=True)

		residual = x
		x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
		# print('src_len:', encoder_out.size())
		# print('padding size:', encoder_padding_mask.size())
		x, attn_weights['enc_attn_weights'] = self.encoder_attn(
			query=x,
			key=encoder_out,
			value=encoder_out,
			key_padding_mask=encoder_padding_mask,
			need_weights=need_enc_attn_weights,
		)
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = residual + x
		x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

		residual = x
		x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
		x = self.activation_fn(self.fc1(x))
		x = self.fc2(x)
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = residual + x
		x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
		return x, attn_weights, past_states

	def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
		assert before ^ after
		if after ^ self.normalize_before:
			return layer_norm(x)
		else:
			return x

class TransformerEncoder(nn.Module):
	"""docstring for TransformerEncoder"""
	def __init__(self, encoder_embed_dim, encoder_ffn_embed_dim, encoder_attention_heads, attention_dropout, dropout, encoder_normalize_before, num_layers):
		super(TransformerEncoder, self).__init__()
		self.layers = nn.ModuleList([TransformerEncoderLayer(
			encoder_embed_dim, encoder_ffn_embed_dim, encoder_attention_heads, 
			attention_dropout, dropout, encoder_normalize_before) for i in range(num_layers)
		])
		self.layer_norm = nn.LayerNorm(encoder_embed_dim) if encoder_normalize_before else None
	def forward(self, x, src_mask=None, encoder_padding_mask=None):
		for layer in self.layers:
			x = layer(x, src_mask, encoder_padding_mask)
		if self.layer_norm is not None:
			x = self.layer_norm(x)
		return x

class TransformerDecoder(nn.Module):
	"""docstring for TransformerDecoder"""
	def __init__(self, decoder_embed_dim, decoder_ffn_embed_dim, decoder_attention_heads, attention_dropout, dropout, decoder_normalize_before, positional_attention, num_layers):
		super(TransformerDecoder, self).__init__()
		self.layers = nn.ModuleList([TransformerDecoderLayer(
			decoder_embed_dim, decoder_ffn_embed_dim, decoder_attention_heads, 
			attention_dropout, dropout, decoder_normalize_before, 
			positional_attention) for i in range(num_layers)
		])
		self.layer_norm = nn.LayerNorm(decoder_embed_dim) if decoder_normalize_before else None
	def forward(self, x,
		encoder_out,
		past_states=None,
		ret_past=False,
		encoder_padding_mask=None,
		self_attn_mask=None,
		position_embedding=None,
		targets_padding=None,
		need_self_attn_weights=False,
		need_pos_attn_weights=False,
		need_enc_attn_weights=False
	):
		attn_weights_all, past_states_all = [], []
		for i, layer in enumerate(self.layers):
			if past_states is None:
				past_self_in, past_pos_in = None, None
			else:
				past_self_in = past_states[i]['self_in']
				past_pos_in = past_states[i]['pos_in'] if 'pos_in' in past_states[i] else None
			
			x, attn_weights, new_past_states = layer(x, encoder_out,
				past_self_in, past_pos_in, ret_past,
				encoder_padding_mask, self_attn_mask,
				position_embedding, targets_padding,
				need_self_attn_weights, need_pos_attn_weights, need_enc_attn_weights
			)
			if need_self_attn_weights or need_pos_attn_weights or need_enc_attn_weights:
				attn_weights_all.append(attn_weights)
			if ret_past:
				past_states_all.append(new_past_states)
		if self.layer_norm is not None:
			x = self.layer_norm(x)
		return x, attn_weights_all, past_states_all
