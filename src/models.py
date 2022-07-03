import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as pad
from layers import *
import datalib.constants as constants
from train_utils import check_values

class style_transfer(nn.Module):
	"""docstring for style_transfer"""
	def __init__(self, vocab_size, emb_size, emb_max_norm,
					rnn_type, enc_hid_size, bidirectional, dec_hid_size, enc_num_layers, dec_num_layers, pooling_size,
					h_only, diff_bias, num_styles, feed_last_context, use_att, enc_cat_style):
		super(style_transfer, self).__init__()
		self.vocab_size = vocab_size
		self.emb_layer = Embedding(vocab_size, emb_size, constants.PAD_ID, emb_max_norm)
		self.enc_cat_style = enc_cat_style
		self.emb_size = emb_size
		self.rnn_type = rnn_type
		self.enc_hid_size = enc_hid_size
		self.bidirectional = bidirectional
		self.enc_num_layers = enc_num_layers
		self.dec_num_layers = dec_num_layers
		self.pooling_size = pooling_size
		self.use_att = use_att
		self.feed_last_context = feed_last_context
		# self.style_direction = style_direction
		self.encoder = getattr(nn, rnn_type)(emb_size, enc_hid_size, num_layers = enc_num_layers, bidirectional = bidirectional)
		enc_size = enc_hid_size * 2 if bidirectional else enc_hid_size
		self.enc_size = enc_size
		self.h_only = h_only
		
		self.num_styles = num_styles
		self.style_emb_layer = Embedding(num_styles, emb_size, max_norm=emb_max_norm)
		# if self.style_direction:
		# 	self.style_direction_layer = nn.Linear(emb_size*2, emb_size)
		self.decoder = attn_decoder(emb_size, dec_hid_size, dec_num_layers, rnn_type, True, enc_size, feed_last_context, use_att=use_att)
		self.out_layer1 = Linear(((enc_size + dec_hid_size) if use_att else dec_hid_size), dec_hid_size)
		self.out_layer2 = multi_bias_linear(num_styles if diff_bias else 1, dec_hid_size, vocab_size)
		# if tie_weights:
		# 	assert emb_size == dec_hid_size
		# 	self.out_layer2.linear.weight = self.emb_layer.weight
		
	def reshape_final_state(self, final_state):
		final_state = final_state.view(self.enc_num_layers, 2 if self.bidirectional else 1, -1, self.enc_hid_size)[-1]
		if self.bidirectional:
			return final_state.transpose(0, 1).contiguous().view(-1, 2*self.enc_hid_size)
		else:
			return final_state.squeeze(0)

	def prepare_dec_init_state(self, enc_last_state):
		init_h = self.reshape_final_state(enc_last_state[0] if self.rnn_type=='LSTM' else enc_last_state)
		init_h = init_h.unsqueeze(0).expand(self.dec_num_layers, init_h.size(0), init_h.size(1)).contiguous()
		if self.rnn_type=='LSTM' and not self.h_only:
			init_c = self.reshape_final_state(enc_last_state[1])
			init_c = init_c.unsqueeze(0).expand(self.dec_num_layers, init_h.size(1), init_h.size(2)).contiguous()
			return (init_h, init_c)
		elif self.rnn_type=='GRU':
			return init_h
		else:
			return (init_h, torch.zeros_like(init_h))

	def get_target_style(self, cur_style):
		if self.num_styles == 2:
			return 1 - cur_style
		else:
			probs = cur_style.new_ones((cur_style.size(0), self.num_styles), dtype = torch.float).scatter_(1, cur_style.view(-1, 1), 0)
			return torch.multinomial(probs, 1).view(-1)

	def embed(self, x, soft=False):
		if soft:
			vocab_vector = torch.arange(0, self.vocab_size, dtype=torch.long, device=x.device)
			x = torch.matmul(x, self.emb_layer(vocab_vector))
		else:
			x = self.emb_layer(x)
		return x

	def encode(self, emb, style_emb, lens, enc_padding_mask, is_sorted=False):
		self.encoder.flatten_parameters()
		if self.enc_cat_style:
			emb = torch.cat([style_emb.unsqueeze(0), emb], 0)

		enc_outputs, final_state = self.encoder(pack(emb, lens.masked_fill(lens==0, 1), enforce_sorted=is_sorted))
		enc_outputs = pad(enc_outputs, total_length=emb.size(0))[0]
		if self.enc_cat_style:
			enc_outputs = enc_outputs[1:]
		
		if self.pooling_size > 1:
			enc_outputs_chunks = enc_outputs.split(self.pooling_size, dim=0)
			enc_padding_mask_chunks = enc_padding_mask.bitwise_not().float().split(self.pooling_size, dim=0)
			enc_outputs_chunks = [c.sum(0, keepdim=True) for c in enc_outputs_chunks]
			enc_padding_mask_chunks = [c.sum(0, keepdim=True) for c in enc_padding_mask_chunks]
			enc_outputs = torch.cat(enc_outputs_chunks, 0)
			enc_padding_mask = torch.cat(enc_padding_mask_chunks, 0)
			enc_outputs = enc_outputs / torch.clamp(enc_padding_mask.unsqueeze(-1), min=1e-10)
			enc_padding_mask = enc_padding_mask == 0

		return enc_outputs, final_state, enc_padding_mask

	def predict_masked_toks(self, states, mask, style_inds):
		masked_states = states.view(-1, states.size(-1))[mask.view(-1)]
		style_inds = style_inds.unsqueeze(0).expand_as(mask)[mask]
		masked_logits = self.out_layer(masked_states, style_inds)
		return masked_logits

	def predict(self, input, last_dec_state, enc_outputs, enc_padding_mask, last_context, style_inds, require_outputs=False, tau=None, greedy=None, 
		less_len_mask=None, exceed_mask=None):
		result = {}
		decoder = self.decoder
		out1, out2 = self.out_layer1, self.out_layer2
		last_dec_state, _, last_context, _ = decoder(input, last_dec_state, enc_outputs, enc_padding_mask, last_context)

		dec_h = last_dec_state[0][-1] if self.rnn_type == 'LSTM' else last_dec_state[-1]
		if not self.use_att:
			logit = out2(out1(dec_h), style_inds)
		else:
			logit = out2(out1(torch.cat([last_context, dec_h], 1)), style_inds)
		
		if less_len_mask is not None:
			logit[less_len_mask, constants.EOS_ID] = -math.inf
		if exceed_mask is not None:
			logit[exceed_mask, constants.EOS_ID] = logit.max().item() + 1000
		result['logits'] = logit
		if require_outputs:
			result['soft_outputs'] = softmax_sample(logit, tau, not greedy)
			result['hard_outputs'] = torch.argmax(result['soft_outputs'], 1)
		return last_dec_state, last_context, result
	
	def para_decode(self, y_emb, style_emb, final_state, enc_outputs, enc_padding_mask, last_context, to_style):
		# enc_outputs, enc_padding_mask, style_emb, last_dec_state, last_context = enc_result
		dec_len = y_emb.size(0) + 1
		result={}
		result['logits'] = torch.zeros(dec_len, y_emb.size(1), self.vocab_size, device=to_style.device)
		
		# dec_emb = self.emb_layer(y)
		# dec_emb = torch.cat([style_emb.unsqueeze(0), dec_emb], 0)
		last_dec_state = self.prepare_dec_init_state(final_state)
		
		for j in range(dec_len):
			input_emb = style_emb if j == 0 else y_emb[j-1]
			last_dec_state, last_context, step_result = self.predict(input_emb, last_dec_state, enc_outputs, enc_padding_mask, 
																				last_context, to_style)
			acc_steps_tensor(result, step_result, j)
		# steps_to_tensor(result)

		return result

	def mono_decode(self, style_emb, final_state, enc_outputs, enc_padding_mask, last_context, to_style, lens, tau, greedy, extra_len, less_len):
		
		result={}
		dec_len = lens.max().item() + extra_len
		lens_no_eos = lens - 1
		result['logits'] = torch.zeros(dec_len, to_style.size(0), self.vocab_size, device=to_style.device)
		result['soft_outputs'] = torch.zeros(dec_len, to_style.size(0), self.vocab_size, device=to_style.device)
		result['hard_outputs'] = torch.zeros(dec_len, to_style.size(0), dtype=torch.long, device=to_style.device)

		inds = torch.arange(0, dec_len, dtype=torch.long, device=style_emb.device).unsqueeze(-1)
		exceed_mask = inds >= (lens_no_eos.unsqueeze(0) + extra_len)
		
		next_emb = style_emb
		last_dec_state = self.prepare_dec_init_state(final_state)
		if less_len is not None:
			least_lens = torch.clamp(lens_no_eos - less_len, min=1)
		
		for j in range(dec_len):
			less_len_mask = (j < least_lens) if less_len is not None else None
			last_dec_state, last_context, step_result = self.predict(next_emb, last_dec_state, enc_outputs, enc_padding_mask, 
																				last_context, to_style, 
																				True, tau, greedy, less_len_mask, exceed_mask[j])
			next_emb = self.emb_layer(step_result['hard_outputs'])
			acc_steps_tensor(result, step_result, j)
		# steps_to_tensor(result)

		# eos_mask = result['hard_outputs'] == constants.EOS_ID
		# result['eos_mask'] = eos_mask
		
		result['hard_outputs_lens'], result['hard_outputs_lens_with_eos'] = get_out_lens(result['hard_outputs'])
		result['hard_outputs_padding_mask'], result['hard_outputs_padding_mask_with_eos'] = get_padding_masks(
			result['hard_outputs'], result['hard_outputs_lens'], result['hard_outputs_lens_with_eos'])
		result['hard_output_zl_mask'] = result['hard_outputs_lens'] == 0
		result['hard_outputs'] = result['hard_outputs'].masked_fill(result['hard_outputs_padding_mask_with_eos'], constants.PAD_ID)
		result['soft_outputs'] = result['soft_outputs'].masked_fill(result['hard_outputs_padding_mask_with_eos'].unsqueeze(-1), 0)
		result['logits'] = result['logits'].masked_fill(result['hard_outputs_padding_mask_with_eos'].unsqueeze(-1), 0)
		return result

	def transfer(self, batch_dict, para, pseudo, simul, mono, enc_pred, with_bt, bt_sg, tau, greedy, extra_len, less_len, beam_decoder=None, same_target=False):
		xc, x, lens, style, seq_mask, px, penc_lens, pseq_mask, cenc_lens, cseq_mask, mx_enc_in, mx_enc_mask = [batch_dict[name] if name in batch_dict else None for name in [
												'xc', 'x', 'enc_lens', 'style', 'seq_mask', 'px', 'penc_lens', 'pseq_mask', 'cenc_lens', 'cseq_mask', 'mx_enc_in', 'mx_enc_mask']]
		seq_emb = self.emb_layer(x)
		seqc_emb = self.emb_layer(xc) if para else None
		pseq_emb = self.emb_layer(px) if pseudo or simul else None
		style_emb = self.style_emb_layer(style)
		to_style = style if same_target else self.get_target_style(style)
		to_style_emb = self.style_emb_layer(to_style)
		
		init_context = torch.zeros(x.size(1), self.enc_size, device=x.device) if self.feed_last_context else None

		para_result, mono_result, pseudo_result, simul_result = None, None, None, None

		if para:
			enc_outputs, final_state, enc_padding_mask = self.encode(seqc_emb, style_emb, cenc_lens, cseq_mask)
			para_result = self.para_decode(seq_emb[:-1], style_emb, final_state, enc_outputs, enc_padding_mask, init_context, style)
		if pseudo:
			enc_outputs, final_state, enc_padding_mask = self.encode(pseq_emb, to_style_emb, penc_lens, pseq_mask)
			pseudo_result = self.para_decode(seq_emb[:-1], style_emb, final_state, enc_outputs, enc_padding_mask, init_context, style)
		if mono or simul:
			mono_result = {}
			mono_result['to_style'] = to_style
			if self.training and mx_enc_in is not None:
				enc_outputs, final_state, enc_padding_mask = self.encode(self.embed(mx_enc_in), style_emb, lens, seq_mask)
				if enc_pred:
					mono_result['enc_logits'] = self.predict_masked_toks(enc_outputs, mx_enc_mask, style)
			else:
				enc_outputs, final_state, enc_padding_mask = self.encode(seq_emb, style_emb, lens, seq_mask)

			if simul:
				simul_result = self.para_decode(pseq_emb[:-1], to_style_emb, final_state, enc_outputs, enc_padding_mask, init_context, to_style)

			if mono:
				if beam_decoder is None:
					fw = self.mono_decode(to_style_emb, final_state, enc_outputs, enc_padding_mask, init_context, to_style, lens, tau, greedy, extra_len, less_len)
				else:
					fw = beam_decoder.generate(False, self, to_style_emb, enc_outputs, enc_padding_mask, to_style, lens, tau, final_state, init_context)
				mono_result['fw'] = fw
				if with_bt:
					enc_outputs, final_state, enc_padding_mask = self.encode(self.embed(fw['hard_outputs' if bt_sg else 'soft_outputs'], not bt_sg), to_style_emb, fw['hard_outputs_lens_with_eos'], fw['hard_outputs_padding_mask_with_eos'], False)
					mono_result['bw'] = self.para_decode(seq_emb[:-1], style_emb, final_state, enc_outputs, enc_padding_mask, init_context, style)
		return para_result, mono_result, pseudo_result, simul_result


class style_transfer_transformer(nn.Module):
	"""docstring for style_transfer"""
	def __init__(self, vocab_size, emb_size, emb_max_norm, pos_sincode, token_emb_scale, pos_max_len, dropout_rate,
					num_heads, hid_size, enc_num_layers, dec_num_layers, diff_bias, num_styles,
					att_dropout_rate, transformer_norm_bf, enc_cat_style, share_pos_emb):
		super(style_transfer_transformer, self).__init__()
		self.vocab_size = vocab_size
		self.token_emb_scale = token_emb_scale
		self.emb_size = emb_size
		self.dropout_rate = dropout_rate
		self.enc_cat_style = enc_cat_style
		self.share_pos_emb = share_pos_emb or pos_sincode
		self.num_styles = num_styles
		
		self.emb_layer = Embedding(vocab_size, emb_size, constants.PAD_ID, emb_max_norm)
		self.enc_pos_emb_layer = PositionalEmbedding(pos_max_len+20 if pos_max_len is not None else None, emb_size, emb_max_norm, not pos_sincode)
		if self.share_pos_emb:
			self.dec_pos_emb_layer = self.enc_pos_emb_layer
		else:
			self.dec_pos_emb_layer = PositionalEmbedding(pos_max_len+20 if pos_max_len is not None else None, emb_size, emb_max_norm, not pos_sincode)
		
		# encoder_layer = nn.TransformerEncoderLayer(emb_size, num_heads, hid_size, dropout_rate)
		# self.encoder = nn.TransformerEncoder(encoder_layer, enc_num_layers)
		self.encoder = TransformerEncoder(emb_size, hid_size, num_heads, att_dropout_rate, dropout_rate, transformer_norm_bf, enc_num_layers)
		# self.style_direction = style_direction
		self.style_emb_layer = Embedding(num_styles, emb_size, max_norm=emb_max_norm)
		# if self.style_direction:
		# 	self.style_direction_layer = Linear(emb_size*2, emb_size)

		# decoder_layer = TransformerDecoderLayer(emb_size, num_heads, hid_size, dropout_rate)
		# self.decoder = TransformerDecoder(decoder_layer, dec_num_layers)
		self.decoder = TransformerDecoder(emb_size, hid_size, num_heads, att_dropout_rate, dropout_rate, transformer_norm_bf, False, dec_num_layers)
		self.out_layer = multi_bias_linear(num_styles if diff_bias else 1, emb_size, vocab_size)
		# if tie_weights:
		# 	self.out_layer.linear.weight = self.emb_layer.weight
		

	def get_target_style(self, cur_style):
		if self.num_styles == 2:
			return 1 - cur_style
		else:
			probs = cur_style.new_ones((cur_style.size(0), self.num_styles), dtype = torch.float).scatter_(1, cur_style.view(-1, 1), 0)
			return torch.multinomial(probs, 1).view(-1)

	def embed(self, x, longest_pos_emb, j=None, soft=False):
		if soft:
			vocab_vector = torch.arange(0, self.vocab_size, dtype=torch.long, device=x.device)
			x = torch.matmul(x, self.emb_layer(vocab_vector))
		else:
			x = self.emb_layer(x)
		if self.token_emb_scale:
			x = x * math.sqrt(self.emb_size)
		# pos_emb_layer = self.enc_pos_emb_layer if is_enc else self.dec_pos_emb_layer
		x = x + (longest_pos_emb[:x.size(0)] if j is None else longest_pos_emb[j])
		
		return x
	def position_embed(self, seq_len, device, is_enc=True):
		pos_emb_layer = self.enc_pos_emb_layer if is_enc else self.dec_pos_emb_layer
		return pos_emb_layer(seq_len, device=device)

	def encode(self, emb, style_emb, enc_padding_mask_b):
		# check_values(x, 'emb', False)
		# if all_zero_mask is not None and all_zero_mask.any():
		# 	enc_padding_mask_b = enc_padding_mask_b.masked_fill(all_zero_mask.unsqueeze(-1), False)
		if self.enc_cat_style:
			emb = torch.cat([style_emb.unsqueeze(0), emb], 0)
			enc_padding_mask_b = torch.cat([enc_padding_mask_b.new_zeros((style_emb.size(0), 1)), enc_padding_mask_b], 1)
		emb = F.dropout(emb, p=self.dropout_rate, training=self.training)
		enc_outputs = self.encoder(emb, encoder_padding_mask=enc_padding_mask_b)
		if self.enc_cat_style:
			enc_outputs = enc_outputs[1:]
		# check_values(enc_outputs, 'enc_outputs', False)
		return enc_outputs
	
	def predict_masked_toks(self, states, mask, style_inds):
		masked_states = states.view(-1, states.size(-1))[mask.view(-1)]
		style_inds = style_inds.unsqueeze(0).expand_as(mask)[mask]
		masked_logits = self.out_layer(masked_states, style_inds)
		return masked_logits

	def seq_predict(self, input, enc_outputs, enc_padding_mask_b, dec_padding_mask_b, style_inds):
		result = {}
		decoder = self.decoder
		out = self.out_layer
		tgt_mask = generate_square_subsequent_mask(input.size(0), input.device)
		dec_h = decoder(input, enc_outputs, self_attn_mask=tgt_mask, targets_padding=dec_padding_mask_b, encoder_padding_mask=enc_padding_mask_b)[0]
		logit = out(dec_h, style_inds)
		result['logits'] = logit
		return result

	def predict(self, input, past, enc_outputs, enc_padding_mask_b, style_inds, require_outputs=False, tau=None, greedy=None, less_len_mask=None, exceed_mask=None):
		result = {}
		decoder = self.decoder
		out = self.out_layer
		dec_h, _, past = decoder(input.unsqueeze(0), enc_outputs, past, True, enc_padding_mask_b)
		logit = out(dec_h[0], style_inds)	
		
		if less_len_mask is not None:
			logit[less_len_mask, constants.EOS_ID] = -math.inf
		if exceed_mask is not None:
			logit[exceed_mask, constants.EOS_ID] = logit.max().item() + 1000
		result['logits'] = logit
		if require_outputs:
			result['soft_outputs'] = softmax_sample(logit, tau, not greedy)
			result['hard_outputs'] = torch.argmax(result['soft_outputs'], -1)
		return past, result

	def para_decode(self, y_emb, style_emb, enc_outputs, enc_padding_mask_b, dec_padding_mask_b, to_style):
		# if all_zero_mask is not None and all_zero_mask.any():
		# 	enc_padding_mask_b = enc_padding_mask_b.masked_fill(all_zero_mask.unsqueeze(-1), False)

		dec_emb = torch.cat([style_emb.unsqueeze(0), y_emb], 0)
		dec_emb = F.dropout(dec_emb, p=self.dropout_rate, training=self.training)
		result = self.seq_predict(dec_emb, enc_outputs, enc_padding_mask_b, dec_padding_mask_b, to_style)

		return result

	def mono_decode(self, style_emb, longest_dec_pos_emb, enc_outputs, enc_padding_mask_b, to_style, lens, tau, greedy, extra_len, less_len):
		
		result={}
		dec_len = lens.max().item() + extra_len
		lens_no_eos = lens - 1
		result['logits'] = torch.zeros(dec_len, to_style.size(0), self.vocab_size, device=to_style.device)
		result['soft_outputs'] = torch.zeros(dec_len, to_style.size(0), self.vocab_size, device=to_style.device)
		result['hard_outputs'] = torch.zeros(dec_len, to_style.size(0), dtype=torch.long, device=to_style.device)

		inds = torch.arange(0, dec_len, dtype=torch.long, device=style_emb.device).unsqueeze(-1)
		exceed_mask = inds >= (lens_no_eos.unsqueeze(0) + extra_len)
		
		next_emb = style_emb#.unsqueeze(0)
		past = None
		if less_len is not None:
			least_lens = torch.clamp(lens_no_eos - less_len, min=1)
		for j in range(dec_len):
			less_len_mask = (j < least_lens) if less_len is not None else None
			next_emb = F.dropout(next_emb, p=self.dropout_rate, training=self.training)
			past, step_result = self.predict(next_emb, past, enc_outputs, enc_padding_mask_b, 
										to_style, True, tau, greedy, less_len_mask, exceed_mask[j])
			next_emb = self.embed(step_result['hard_outputs'], longest_dec_pos_emb, j=j)#.unsqueeze(0)
			acc_steps_tensor(result, step_result, j)
			
		
		result['hard_outputs_lens'], result['hard_outputs_lens_with_eos'] = get_out_lens(result['hard_outputs'])
		result['hard_outputs_padding_mask'], result['hard_outputs_padding_mask_with_eos'] = get_padding_masks(
			result['hard_outputs'], result['hard_outputs_lens'], result['hard_outputs_lens_with_eos'])
		result['hard_output_zl_mask'] = result['hard_outputs_lens'] == 0

		result['hard_outputs_padding_mask_t'] = result['hard_outputs_padding_mask'].t().contiguous()
		result['hard_outputs_padding_mask_t_with_eos'] = result['hard_outputs_padding_mask_with_eos'].t().contiguous()
		# result['hard_outputs_padding_mask_t'].masked_fill_(result['hard_output_zl_mask'].unsqueeze(1), False)
		# print('num of zero len:',result['hard_output_zl_mask'].sum().item())
		result['hard_outputs'] = result['hard_outputs'].masked_fill(result['hard_outputs_padding_mask_with_eos'], constants.PAD_ID)
		result['soft_outputs'] = result['soft_outputs'].masked_fill(result['hard_outputs_padding_mask_with_eos'].unsqueeze(-1), 0)
		result['logits'] = result['logits'].masked_fill(result['hard_outputs_padding_mask_with_eos'].unsqueeze(-1), 0)
		return result

	def transfer(self, batch_dict, para, pseudo, simul, mono, enc_pred, with_bt, bt_sg, tau, greedy, extra_len, less_len, beam_decoder=None, same_target=False):
		xc, x, lens, style, seq_mask_b, px, pseq_mask_b, cseq_mask_b, mx_enc_in, mx_enc_mask = [batch_dict[name] if name in batch_dict else None for name in [
							'xc', 'x', 'enc_lens', 'style', 'seq_mask_b', 'px', 'pseq_mask_b', 'cseq_mask_b', 'mx_enc_in', 'mx_enc_mask']]

		style_emb = self.style_emb_layer(style)
		to_style = style if same_target else self.get_target_style(style)
		to_style_emb = self.style_emb_layer(to_style)

		bsz = style.size(0)
		# print(max([v.size(0) for v in [xc, x, px, mx_enc_in] if v is not None]))
		# print(lens.max().item()+extra_len)
		max_positions = max(max([v.size(0) for v in [xc, x, px, mx_enc_in] if v is not None]), lens.max().item()+extra_len) + 1
		longest_enc_pos_emb = self.position_embed(max_positions, style.device, True).expand(max_positions, bsz, self.emb_size).contiguous()
		longest_dec_pos_emb = longest_enc_pos_emb if self.share_pos_emb else self.position_embed(max_positions, style.device, False).expand(max_positions, bsz, self.emb_size).contiguous()
		style_enc_pos_emb, style_dec_pos_emb, longest_enc_pos_emb, longest_dec_pos_emb = longest_enc_pos_emb[0], longest_dec_pos_emb[0], longest_enc_pos_emb[1:], longest_dec_pos_emb[1:]
		style_enc_emb = style_emb + style_enc_pos_emb
		to_style_enc_emb = to_style_emb + style_enc_pos_emb
		style_dec_emb = style_emb + style_dec_pos_emb
		to_style_dec_emb = to_style_emb + style_dec_pos_emb

		enc_seq_emb = self.embed(x, longest_enc_pos_emb) if (mono or simul) and not (self.training and mx_enc_in is not None) else None
		if para or pseudo or (mono and with_bt):
			dec_seq_emb = enc_seq_emb[:-1] if (enc_seq_emb is not None) and self.share_pos_emb else self.embed(x[:-1], longest_dec_pos_emb)
		else:
			dec_seq_emb = None

		para_result, mono_result, pseudo_result, simul_result = None, None, None, None
		if para:
			para_result = self.para_decode(dec_seq_emb, style_dec_emb, self.encode(self.embed(xc, longest_enc_pos_emb), style_enc_emb, cseq_mask_b), 
				cseq_mask_b, seq_mask_b, style)
		if pseudo:
			pseudo_result = self.para_decode(dec_seq_emb, style_dec_emb, self.encode(self.embed(px, longest_enc_pos_emb), to_style_enc_emb, pseq_mask_b), 
				pseq_mask_b, seq_mask_b, style)
		if mono or simul:
			mono_result = {}
			mono_result['to_style'] = to_style
			if self.training and mx_enc_in is not None:
				enc_outputs = self.encode(self.embed(mx_enc_in, longest_enc_pos_emb), style_enc_emb, seq_mask_b)
				if enc_pred:
					mono_result['enc_logits'] = self.predict_masked_toks(enc_outputs, mx_enc_mask, style)
			else:
				enc_outputs = self.encode(enc_seq_emb, style_enc_emb, seq_mask_b)

			if simul:
				simul_result = self.para_decode(self.embed(px[:-1], longest_dec_pos_emb), to_style_dec_emb, enc_outputs, seq_mask_b, pseq_mask_b, to_style)

			if mono:
				if beam_decoder is None:
					fw = self.mono_decode(to_style_dec_emb, longest_dec_pos_emb, enc_outputs, seq_mask_b, to_style, lens, tau, greedy, extra_len, less_len)
				else:
					fw = beam_decoder.generate(True, self, to_style_dec_emb, longest_dec_pos_emb, enc_outputs, seq_mask_b, to_style, lens, tau)
				mono_result['fw'] = fw
				if with_bt:
					mono_result['bw'] = self.para_decode(dec_seq_emb, style_dec_emb, 
						self.encode(self.embed(fw['hard_outputs' if bt_sg else 'soft_outputs'], longest_enc_pos_emb, soft=not bt_sg), to_style_enc_emb, fw['hard_outputs_padding_mask_t_with_eos']), 
						fw['hard_outputs_padding_mask_t_with_eos'], seq_mask_b, style)
		return para_result, mono_result, pseudo_result, simul_result

	



