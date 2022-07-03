import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
import datalib.constants as constants
from rescore import *
from loss import coverage_score

class nat_style_transfer(nn.Module):
	"""docstring for nat_style_transfer"""
	def __init__(self, vocab_size, emb_size, emb_max_norm, pos_sincode, token_emb_scale, pos_max_len, dropout_rate,
					num_heads, hid_size, enc_num_layers, dec_num_layers, diff_bias, num_styles, 
					att_dropout_rate, transformer_norm_bf, enc_cat_style, share_pos_emb, 
					positional_att, apply_self_mask, self_mask_escape_start):
		super(nat_style_transfer, self).__init__()
		self.vocab_size = vocab_size
		self.token_emb_scale = token_emb_scale
		self.share_pos_emb = share_pos_emb or pos_sincode
		self.emb_size = emb_size
		self.dropout_rate = dropout_rate
		self.enc_cat_style = enc_cat_style
		self.num_styles = num_styles
		self.positional_att = positional_att
		self.apply_self_mask = apply_self_mask
		self.self_mask_escape_start = self_mask_escape_start
		
		self.emb_layer = Embedding(vocab_size, emb_size, constants.PAD_ID, emb_max_norm)
		# an extra position for the style
		self.enc_pos_emb_layer = PositionalEmbedding(pos_max_len+20 if pos_max_len is not None else None, emb_size, emb_max_norm, not pos_sincode)
		if self.share_pos_emb:
			self.dec_pos_emb_layer = self.enc_pos_emb_layer
		else:
			self.dec_pos_emb_layer = PositionalEmbedding(pos_max_len+20 if pos_max_len is not None else None, emb_size, emb_max_norm, not pos_sincode)
		self.style_emb_layer = Embedding(num_styles, emb_size, max_norm=emb_max_norm)
		
		self.encoder = TransformerEncoder(emb_size, hid_size, num_heads, att_dropout_rate, dropout_rate, transformer_norm_bf, enc_num_layers)
		self.decoder = TransformerDecoder(emb_size, hid_size, num_heads, att_dropout_rate, dropout_rate, transformer_norm_bf, positional_att, dec_num_layers)
		self.out_layer = multi_bias_linear(num_styles if diff_bias else 1, emb_size, vocab_size)

	def get_target_style(self, cur_style):
		if self.num_styles == 2:
			return 1 - cur_style
		else:
			probs = cur_style.new_ones((cur_style.size(0), self.num_styles), dtype = torch.float).scatter_(1, cur_style.view(-1, 1), 0)
			return torch.multinomial(probs, 1).view(-1)
	
	def token_embed(self, x, soft=False):
		if soft:
			vocab_vector = torch.arange(0, self.vocab_size, dtype=torch.long, device=x.device)
			x = torch.matmul(x, self.emb_layer(vocab_vector))
		else:
			x = self.emb_layer(x)
		if self.token_emb_scale:
			x = x * math.sqrt(self.emb_size)
		return x
	
	def position_embed(self, seq_len, device, is_enc=True):
		pos_emb_layer = self.enc_pos_emb_layer if is_enc else self.dec_pos_emb_layer
		return pos_emb_layer(seq_len, device=device)

	def embed(self, x, longest_pos_emb, soft=False):
		return self.token_embed(x, soft) + longest_pos_emb[:x.size(0)]

	def add_tok_and_pos(self, tok_emb, pos_emb):
		return tok_emb + pos_emb[:tok_emb.size(0)]
	
	def encode(self, emb, style_emb, enc_padding_mask_b, mx_mask=None, style_inds=None):
		
		if self.enc_cat_style:
			emb = torch.cat([style_emb.unsqueeze(0), emb], 0)
			enc_padding_mask_b = torch.cat([enc_padding_mask_b.new_zeros((style_emb.size(0), 1)), enc_padding_mask_b], 1)
		emb = F.dropout(emb, p=self.dropout_rate, training=self.training)
		enc_outputs = self.encoder(emb, encoder_padding_mask=enc_padding_mask_b)
		if self.enc_cat_style:
			enc_outputs = enc_outputs[1:]

		if mx_mask is not None:
			masked_logits = self.predict_masked_toks(enc_outputs, mx_mask, style_inds)
		else:
			masked_logits = None
		return enc_outputs, masked_logits

	def predict_masked_toks(self, states, mask, style_inds):
		if mask is not None:
			states = states.view(-1, states.size(-1))[mask.view(-1)]
			style_inds = style_inds.unsqueeze(0).expand_as(mask)[mask]
		logits = self.out_layer(states, style_inds)
		return logits
		
	def decode(self, input_emb, style_emb, longest_pos_emb, pos_emb, enc_outputs, enc_padding_mask_b, dec_padding_mask_b, mx_mask, to_style, 
		need_self_attn=False, need_pos_attn=False, need_enc_attn=False, require_outputs=False, tau=None, greedy=None,
		fill_eos=False, dec_lens=None, recompute_lens=False, less_len=None):
		result = {}
		emb = torch.cat([style_emb.unsqueeze(0), input_emb], 0)
		emb = F.dropout(emb, p=self.dropout_rate, training=self.training)
		self_attn_mask = generate_self_mask(emb.size(0), emb.device, self.self_mask_escape_start) if self.apply_self_mask else None
		bsz = style_emb.size(0)
		dec_padding_mask_b_ext = torch.cat([dec_padding_mask_b.new_zeros((bsz, 1)), dec_padding_mask_b], 1)
		if pos_emb is None:
			pos_emb = longest_pos_emb[:emb.size(0)]
		else:
			pos_emb = torch.cat([longest_pos_emb[:1], pos_emb], 0)
		states, result['attn_weights'], _ = self.decoder(emb, enc_outputs, None, False, enc_padding_mask_b, self_attn_mask, pos_emb, dec_padding_mask_b_ext,
			need_self_attn, need_pos_attn, need_enc_attn)
		states = states[1:]
		
		result['logits'] = self.predict_masked_toks(states, mx_mask, to_style)
		dec_lens_no_eos = dec_lens - 1 if dec_lens is not None else None
		if recompute_lens and less_len is not None:
			seq_len = states.size(0)
			least_lens = torch.clamp(dec_lens_no_eos - less_len, min=1)
			less_len_mask = get_padding_mask_on_size(seq_len, least_lens, flip=True)
			result['logits'].view(-1, self.vocab_size)[less_len_mask.view(-1), constants.EOS_ID] = -math.inf

		if fill_eos:
			# assert mx_mask is None
			fvalue = result['logits'].max().item() + 1
			binds = torch.arange(0, bsz, dtype=torch.long, device=to_style.device)
			result['logits'][dec_lens_no_eos, binds, constants.EOS_ID] = fvalue

		if require_outputs:
			result['soft_outputs'] = softmax_sample(result['logits'], tau, not greedy)
			result['hard_outputs'] = torch.argmax(result['soft_outputs'], -1)
			if fill_eos:
				# assert mx_mask is None
				# result['hard_outputs'].scatter_(0, dec_lens_no_eos.unsqueeze(0), constants.EOS_ID)
				if recompute_lens:
					result['hard_outputs_lens'], result['hard_outputs_lens_with_eos'] = get_out_lens(result['hard_outputs'], return_with_eos=True)
					result['hard_outputs_padding_mask'], result['hard_outputs_padding_mask_with_eos'] = get_padding_masks(
						result['hard_outputs'], result['hard_outputs_lens'], result['hard_outputs_lens_with_eos'])
					result['hard_output_zl_mask'] = result['hard_outputs_lens'] == 0
					result['hard_outputs_padding_mask_t'] = result['hard_outputs_padding_mask'].t().contiguous()
					result['hard_outputs_padding_mask_t_with_eos'] = result['hard_outputs_padding_mask_with_eos'].t().contiguous()
				else:
					result['hard_outputs_lens'] = dec_lens_no_eos
					result['hard_outputs_lens_with_eos'] = dec_lens
					result['hard_outputs_padding_mask_t_with_eos'] = dec_padding_mask_b
					result['hard_outputs_padding_mask_with_eos'] = dec_padding_mask_b.t().contiguous()



			# if mx_mask is None:
			# 	mask = result['hard_outputs_padding_mask_with_eos'] if fill_eos else 
			# 	result['hard_outputs'] = result['hard_outputs'].masked_fill(dec_padding_mask_b.t(), constants.PAD_ID)
			# 	result['soft_outputs'] = result['soft_outputs'].masked_fill(dec_padding_mask_b.t().unsqueeze(-1), 0)
		return result

	def transfer(self, batch_dict, arg_dict):
		# print('new batch')
		style, xc, clens, cseq_mask_b, x, lens, seq_mask, seq_mask_b, px, plens, pseq_mask, pseq_mask_b, mx_enc_in, mx_enc_mask, mx_dec_in, mx_dec_mask, mpx_dec_in, mpx_dec_mask = [
				batch_dict.get(name, None) for name in [
				'style', 'xc', 'clens', 'cseq_mask_b', 
				'x', 'lens', 'seq_mask', 'seq_mask_b', 
				'px', 'plens', 'pseq_mask', 'pseq_mask_b', 
				'mx_enc_in', 'mx_enc_mask', 'mx_dec_in', 'mx_dec_mask', 
				'mpx_dec_in', 'mpx_dec_mask']]
		device = style.device
		style_emb = self.style_emb_layer(style)
		to_style = self.get_target_style(style)
		to_style_emb = self.style_emb_layer(to_style)
		bsz = style.size(0)
		# need an extra position for the style
		max_positions = (max([v.size(0) for v in [xc, x, px, mx_enc_in, mx_dec_in, mpx_dec_in] if v is not None])
					+5+(arg_dict['ctr_kmax'] if arg_dict['ctr_kmax'] is not None else 0) + 1)
		longest_enc_pos_emb = self.position_embed(max_positions, style.device, True).expand(max_positions, bsz, self.emb_size).contiguous()
		longest_dec_pos_emb = longest_enc_pos_emb if self.share_pos_emb else self.position_embed(max_positions, style.device, False).expand(max_positions, bsz, self.emb_size).contiguous()
		longest_dec_pos_emb_full = longest_dec_pos_emb
		style_enc_pos_emb, style_dec_pos_emb, longest_enc_pos_emb, longest_dec_pos_emb = longest_enc_pos_emb[0], longest_dec_pos_emb[0], longest_enc_pos_emb[1:], longest_dec_pos_emb[1:]
		style_enc_emb = style_emb + style_enc_pos_emb
		to_style_enc_emb = to_style_emb + style_enc_pos_emb
		style_dec_emb = style_emb + style_dec_pos_emb
		to_style_dec_emb = to_style_emb + style_dec_pos_emb
		
		if (arg_dict['rec'] or arg_dict['pseudo'] or arg_dict['bt'] or arg_dict['mpx_bt'] or arg_dict['fc_bt']) and arg_dict['dec_mask_mode']:
			# para_x_dec_emb = self.embed(mask_input(x, arg_dict['fm_fill_eos']) if arg_dict['para_x_fm'] else mx_dec_in, longest_dec_pos_emb)
			para_x_dec_tok = mask_input(x, arg_dict['fm_fill_eos']) if arg_dict['para_x_fm'] else mx_dec_in
			para_x_dec_tok_emb = self.token_embed(para_x_dec_tok)
			para_x_dec_emb = self.add_tok_and_pos(para_x_dec_tok_emb, longest_dec_pos_emb)

		result = {}
		result['to_style'] = to_style
		if arg_dict['rec']:
			xc_emb = self.embed(xc, longest_enc_pos_emb)
			enc_outputs, _ = self.encode(xc_emb, style_enc_emb, cseq_mask_b)
			# dec_input_emb = para_x_dec_emb if arg_dict['dec_mask_mode'] else self.embed(uniform_copy(xc, clens, lens, seq_mask), longest_dec_pos_emb)
			if arg_dict['dec_mask_mode']:
				dec_input_tok = para_x_dec_tok
				dec_input_tok_emb = para_x_dec_tok_emb
				dec_input_emb = para_x_dec_emb
			else:
				dec_input_tok = uniform_copy(xc, clens, lens, seq_mask)
				dec_input_tok_emb = self.token_embed(dec_input_tok)
				dec_input_emb = self.add_tok_and_pos(dec_input_tok_emb, longest_dec_pos_emb)

			result['rec'] = self.decode(dec_input_emb, style_dec_emb, longest_dec_pos_emb_full, None, enc_outputs, cseq_mask_b, seq_mask_b,
				mx_dec_mask, style)
			if arg_dict['rec_dctr']:
				result['rec_dctr'] = []
				result['rec_dctr_word_ims'] = []
				result['rec_dctr_unchange'] = []
				result['rec_dctr_masks'] = []
				neg_samples = generate_negative_samples_by_delete(arg_dict['ctr_n'], seq_mask, mx_dec_mask, lens, arg_dict['ctr_kmin'], arg_dict['ctr_kmax'], arg_dict['ctr_word_im'])
				for sample in neg_samples:
					ctr_padding_mask, ctr_mx_mask, ctr_positions, ctr_word_ims = sample
					ctr_lens = (~ctr_padding_mask).long().sum(0)
					pos_emb = self.dec_pos_emb_layer(device=device, detail_pos=ctr_positions)
					ctr_input_emb = self.add_tok_and_pos(dec_input_tok_emb, pos_emb)
					result['rec_dctr'].append(self.decode(ctr_input_emb, style_dec_emb, longest_dec_pos_emb_full, pos_emb, 
						enc_outputs, cseq_mask_b, ctr_padding_mask.t(), ctr_mx_mask, style)['logits'])
					result['rec_dctr_word_ims'].append(ctr_word_ims)
					result['rec_dctr_unchange'].append(ctr_lens==lens)
					result['rec_dctr_masks'].append(ctr_padding_mask if ctr_mx_mask is None else ctr_mx_mask)
			if arg_dict['rec_ictr']:
				result['rec_ictr'] = []
				result['rec_ictr_word_ims'] = []
				result['rec_ictr_masks'] = []
				result['rec_ictr_filling_masks'] = []
				neg_samples = generate_negative_samples_by_insert(arg_dict['ctr_n'], dec_input_tok, seq_mask, lens, arg_dict['ctr_kmin'], arg_dict['ctr_kmax'], arg_dict['ctr_word_im'])
				for sample in neg_samples:
					ctr_input, ctr_padding_mask, ctr_mx_mask, ctr_word_ims, ctr_filling_mask = sample
					ctr_input_emb = self.embed(ctr_input, longest_dec_pos_emb)
					result['rec_ictr'].append(self.decode(ctr_input_emb, style_dec_emb, longest_dec_pos_emb_full, None, 
						enc_outputs, cseq_mask_b, ctr_padding_mask.t(), ctr_mx_mask if mx_dec_mask is not None else None, style)['logits'])
					result['rec_ictr_word_ims'].append(ctr_word_ims)
					result['rec_ictr_masks'].append(ctr_padding_mask if mx_dec_mask is None else ctr_mx_mask)
					result['rec_ictr_filling_masks'].append(ctr_filling_mask)
			
		
		if arg_dict['pseudo']:
			px_emb = self.embed(px, longest_enc_pos_emb)
			enc_outputs, _ = self.encode(px_emb, to_style_enc_emb, pseq_mask_b)
			dec_input_emb = para_x_dec_emb if arg_dict['dec_mask_mode'] else self.embed(uniform_copy(px, plens, lens, seq_mask), longest_dec_pos_emb)
			result['pseudo'] = self.decode(dec_input_emb, style_dec_emb, longest_dec_pos_emb_full, None, enc_outputs, pseq_mask_b, seq_mask_b,
				mx_dec_mask, style)
		
		if arg_dict['dec_use_px'] or arg_dict['dec_use_mpx'] or arg_dict['dec_use_fm'] or arg_dict['dec_use_fc'] or arg_dict['fc_bt'] or arg_dict['enc_pred']:
			if arg_dict['enc_pred']:
				assert not arg_dict['enc_use_x']
			enc_input_emb = self.embed(x if arg_dict['enc_use_x'] else mx_enc_in, longest_enc_pos_emb)
			enc_outputs, result['enc_logits'] = self.encode(enc_input_emb, style_enc_emb, seq_mask_b, mx_enc_mask if arg_dict['enc_pred'] else None, style)
			if arg_dict['dec_use_px']:
				# dec_input_emb = self.embed(mask_input(px, arg_dict['fm_fill_eos']) if arg_dict['dec_mask_mode'] else uniform_copy(x, lens, plens, pseq_mask_b.t()), longest_dec_pos_emb)
				if arg_dict['dec_mask_mode']:
					dec_input_tok = mask_input(px, arg_dict['fm_fill_eos'])
				else:
					dec_input_tok = uniform_copy(x, lens, plens, pseq_mask)
				dec_input_tok_emb = self.token_embed(dec_input_tok)
				dec_input_emb = self.add_tok_and_pos(dec_input_tok_emb, longest_dec_pos_emb)
				result['dec_px'] = self.decode(dec_input_emb, to_style_dec_emb, longest_dec_pos_emb_full, None, enc_outputs, seq_mask_b, pseq_mask_b, None,
					to_style, arg_dict['need_self_attn'], False, arg_dict['need_enc_attn'])
				if arg_dict['px_dctr']:
					result['px_dctr'] = []
					result['px_dctr_word_ims'] = []
					result['px_dctr_unchange'] = []
					result['px_dctr_masks'] = []
					neg_samples = generate_negative_samples_by_delete(arg_dict['ctr_n'], pseq_mask, None, plens, arg_dict['ctr_kmin'], arg_dict['ctr_kmax'], arg_dict['ctr_word_im'])
					for sample in neg_samples:
						ctr_padding_mask, ctr_mx_mask, ctr_positions, ctr_word_ims = sample
						ctr_lens = (~ctr_padding_mask).long().sum(0)
						pos_emb = self.dec_pos_emb_layer(device=device, detail_pos=ctr_positions)
						ctr_input_emb = self.add_tok_and_pos(dec_input_tok_emb, pos_emb)
						result['px_dctr'].append(self.decode(ctr_input_emb, to_style_dec_emb, longest_dec_pos_emb_full, pos_emb, 
							enc_outputs, seq_mask_b, ctr_padding_mask.t(), None, to_style)['logits'])
						result['px_dctr_word_ims'].append(ctr_word_ims)
						result['px_dctr_unchange'].append(ctr_lens==plens)
						result['px_dctr_masks'].append(ctr_padding_mask)
				if arg_dict['px_ictr']:
					result['px_ictr'] = []
					result['px_ictr_word_ims'] = []
					result['px_ictr_masks'] = []
					result['px_ictr_filling_masks'] = []
					neg_samples = generate_negative_samples_by_insert(arg_dict['ctr_n'], dec_input_tok, pseq_mask, plens, arg_dict['ctr_kmin'], arg_dict['ctr_kmax'], arg_dict['ctr_word_im'])
					for sample in neg_samples:
						ctr_input, ctr_padding_mask, ctr_mx_mask, ctr_word_ims, ctr_filling_mask = sample
						ctr_input_emb = self.embed(ctr_input, longest_dec_pos_emb)
						result['px_ictr'].append(self.decode(ctr_input_emb, to_style_dec_emb, longest_dec_pos_emb_full, None, 
							enc_outputs, seq_mask_b, ctr_padding_mask.t(), None, to_style)['logits'])
						result['px_ictr_word_ims'].append(ctr_word_ims)
						result['px_ictr_masks'].append(ctr_padding_mask)
						result['px_ictr_filling_masks'].append(ctr_filling_mask)
			if arg_dict['dec_use_mpx']:
				# mpx_emb = self.embed(mpx_dec_in, longest_dec_pos_emb)
				mpx_tok_emb = self.token_embed(mpx_dec_in)
				mpx_emb = self.add_tok_and_pos(mpx_tok_emb, longest_dec_pos_emb)
				result['dec_mpx'] = self.decode(mpx_emb, to_style_dec_emb, longest_dec_pos_emb_full, None, enc_outputs, seq_mask_b, pseq_mask_b, None,
					to_style, arg_dict['need_self_attn'], False, arg_dict['need_enc_attn'], arg_dict['mpx_out'], arg_dict['tau'], arg_dict['greedy'])
				if arg_dict['mpx_dctr']:
					result['mpx_dctr'] = []
					result['mpx_dctr_word_ims'] = []
					result['mpx_dctr_unchange'] = []
					result['mpx_dctr_masks'] = []
					neg_samples = generate_negative_samples_by_delete(arg_dict['ctr_n'], pseq_mask, mpx_dec_mask, plens, arg_dict['ctr_kmin'], arg_dict['ctr_kmax'], arg_dict['ctr_word_im'])
					for sample in neg_samples:
						ctr_padding_mask, ctr_mx_mask, ctr_positions, ctr_word_ims = sample
						ctr_lens = (~ctr_padding_mask).long().sum(0)
						pos_emb = self.dec_pos_emb_layer(device=device, detail_pos=ctr_positions)
						ctr_input_emb = self.add_tok_and_pos(mpx_tok_emb, pos_emb)
						result['mpx_dctr'].append(self.decode(ctr_input_emb, to_style_dec_emb, longest_dec_pos_emb_full, pos_emb, 
							enc_outputs, seq_mask_b, ctr_padding_mask.t(), None, to_style)['logits'])
						result['mpx_dctr_word_ims'].append(ctr_word_ims)
						result['mpx_dctr_unchange'].append(ctr_lens==plens)
						result['mpx_dctr_masks'].append(ctr_mx_mask)
				if arg_dict['mpx_ictr']:
					result['mpx_ictr'] = []
					result['mpx_ictr_word_ims'] = []
					result['mpx_ictr_masks'] = []
					result['mpx_ictr_filling_masks'] = []
					neg_samples = generate_negative_samples_by_insert(arg_dict['ctr_n'], mpx_dec_in, pseq_mask, plens, arg_dict['ctr_kmin'], arg_dict['ctr_kmax'], arg_dict['ctr_word_im'])
					for sample in neg_samples:
						ctr_input, ctr_padding_mask, ctr_mx_mask, ctr_word_ims, ctr_filling_mask = sample
						ctr_input_emb = self.embed(ctr_input, longest_dec_pos_emb)
						result['mpx_ictr'].append(self.decode(ctr_input_emb, to_style_dec_emb, longest_dec_pos_emb_full, None, 
							enc_outputs, seq_mask_b, ctr_padding_mask.t(), None, to_style)['logits'])
						result['mpx_ictr_word_ims'].append(ctr_word_ims)
						result['mpx_ictr_masks'].append(ctr_mx_mask)
						result['mpx_ictr_filling_masks'].append(ctr_filling_mask)


				if arg_dict['mpx_bt']:
					mpx_bt_input = mpx_dec_in.masked_scatter(mpx_dec_mask, result['dec_mpx']['hard_outputs'][mpx_dec_mask])
					bt_emb = self.embed(mpx_bt_input, longest_enc_pos_emb)
					mpx_enc_outputs, _ = self.encode(bt_emb, to_style_enc_emb, pseq_mask_b)
					result['mpx_bt'] = self.decode(para_x_dec_emb, style_dec_emb, longest_dec_pos_emb_full, None, mpx_enc_outputs, pseq_mask_b, seq_mask_b,
						mx_dec_mask, style)
					if arg_dict['mpx_bt_dctr']:
						result['mpx_bt_dctr'] = []
						result['mpx_bt_dctr_word_ims'] = []
						result['mpx_bt_dctr_unchange'] = []
						result['mpx_bt_dctr_masks'] = []
						neg_samples = generate_negative_samples_by_delete(arg_dict['ctr_n'], seq_mask, mx_dec_mask, lens, arg_dict['ctr_kmin'], arg_dict['ctr_kmax'], arg_dict['ctr_word_im'])
						for sample in neg_samples:
							ctr_padding_mask, ctr_mx_mask, ctr_positions, ctr_word_ims = sample
							ctr_lens = (~ctr_padding_mask).long().sum(0)
							pos_emb = self.dec_pos_emb_layer(device=device, detail_pos=ctr_positions)
							ctr_input_emb = self.add_tok_and_pos(para_x_dec_tok_emb, pos_emb)
							result['mpx_bt_dctr'].append(self.decode(ctr_input_emb, style_dec_emb, longest_dec_pos_emb_full, pos_emb, 
								mpx_enc_outputs, pseq_mask_b, ctr_padding_mask.t(), ctr_mx_mask, style)['logits'])
							result['mpx_bt_dctr_word_ims'].append(ctr_word_ims)
							result['mpx_bt_dctr_unchange'].append(ctr_lens==lens)
							result['mpx_bt_dctr_masks'].append(ctr_padding_mask if ctr_mx_mask is None else ctr_mx_mask)
					if arg_dict['mpx_bt_ictr']:
						result['mpx_bt_ictr'] = []
						result['mpx_bt_ictr_word_ims'] = []
						result['mpx_bt_ictr_masks'] = []
						result['mpx_bt_ictr_filling_masks'] = []
						neg_samples = generate_negative_samples_by_insert(arg_dict['ctr_n'], para_x_dec_tok, seq_mask, lens, arg_dict['ctr_kmin'], arg_dict['ctr_kmax'], arg_dict['ctr_word_im'])
						for sample in neg_samples:
							ctr_input, ctr_padding_mask, ctr_mx_mask, ctr_word_ims, ctr_filling_mask = sample
							ctr_input_emb = self.embed(ctr_input, longest_dec_pos_emb)
							result['mpx_bt_ictr'].append(self.decode(ctr_input_emb, style_dec_emb, longest_dec_pos_emb_full, None, 
								mpx_enc_outputs, pseq_mask_b, ctr_padding_mask.t(), ctr_mx_mask if mx_dec_mask is not None else None, style)['logits'])
							result['mpx_bt_ictr_word_ims'].append(ctr_word_ims)
							result['mpx_bt_ictr_masks'].append(ctr_padding_mask if mx_dec_mask is None else ctr_mx_mask)
							result['mpx_bt_ictr_filling_masks'].append(ctr_filling_mask)

				if arg_dict['dec_use_mpx_tch']:
					with torch.set_grad_enabled(arg_dict['tch_sym']):
						mpx_dec_mask_tch = mpx_dec_mask.new_empty(mpx_dec_mask.size()).bernoulli_(arg_dict['tch_keep_rate']) & mpx_dec_mask
						while (~mpx_dec_mask_tch).all():
							mpx_dec_mask_tch = mpx_dec_mask.new_empty(mpx_dec_mask.size()).bernoulli_(arg_dict['tch_keep_rate']) & mpx_dec_mask
						mpx_dec_in_tch = px.masked_fill(mpx_dec_mask_tch, constants.MSK_ID)
						mpx_emb_tch = self.embed(mpx_dec_in_tch, longest_dec_pos_emb)
						result['mpx_dec_mask_tch'] = mpx_dec_mask_tch
						result['dec_mpx_tch'] = self.decode(mpx_emb_tch, to_style_dec_emb, longest_dec_pos_emb_full, None, enc_outputs, seq_mask_b, pseq_mask_b, mpx_dec_mask_tch,
							to_style, arg_dict['need_self_attn'], False, arg_dict['need_enc_attn'])

				if arg_dict['dec_use_mpx_iter']:
					with torch.set_grad_enabled(arg_dict['iter_sym']):
						mpx_dec_in_iter, mpx_dec_mask_iter = replace_toks(result['dec_mpx']['hard_outputs'], result['dec_mpx']['soft_outputs'],
							mpx_dec_in, mpx_dec_mask, arg_dict['iter_keep_rate'], arg_dict['iter_random'])
						while (~mpx_dec_mask_iter).all():
							mpx_dec_in_iter, mpx_dec_mask_iter = replace_toks(result['dec_mpx']['hard_outputs'], result['dec_mpx']['soft_outputs'],
								mpx_dec_in, mpx_dec_mask, arg_dict['iter_keep_rate'], arg_dict['iter_random'])
						mpx_emb_iter = self.embed(mpx_dec_in_iter, longest_dec_pos_emb)
						result['mpx_dec_mask_iter'] = mpx_dec_mask_iter
						result['dec_mpx_iter'] = self.decode(mpx_emb_iter, to_style_dec_emb, longest_dec_pos_emb_full, None, enc_outputs, seq_mask_b, pseq_mask_b,
							mpx_dec_mask_iter, to_style, arg_dict['need_self_attn'], False, arg_dict['need_enc_attn'])
			if arg_dict['dec_use_fm']:
				dec_len = max((px.size(0) if arg_dict['fm_use_pseudo_len'] else x.size(0)) + arg_dict['extra_len'], 2)
				fm_lens = torch.clamp((plens if arg_dict['fm_use_pseudo_len'] else lens) + arg_dict['extra_len'], min=2)
				fm_padding_mask = get_padding_mask_on_size(dec_len, fm_lens)
				fm_padding_mask_b = fm_padding_mask.t().contiguous()
				if arg_dict['dec_mask_mode']:
					full_mask = x.new_full((dec_len, bsz), constants.MSK_ID).masked_fill_(fm_padding_mask, constants.PAD_ID)
					if arg_dict['fm_fill_eos']:
						full_mask.scatter_(0, fm_lens.unsqueeze(0)-1, constants.EOS_ID)
					fm_emb = self.embed(full_mask, longest_dec_pos_emb)
				else:
					fm_emb = self.embed(uniform_copy(x, lens, fm_lens, fm_padding_mask), longest_dec_pos_emb)
				result['fm_lens'] = fm_lens
				result['fm_padding_mask'] = fm_padding_mask
				result['fm_padding_mask_b'] = fm_padding_mask_b
				# print('x size:', x.size())
				# print('enc size:', enc_outputs.size())
				# print('padding mask:', seq_mask_b.size())
				result['dec_fm'] = self.decode(fm_emb, to_style_dec_emb, longest_dec_pos_emb_full, None, enc_outputs, seq_mask_b, fm_padding_mask_b, None, 
					to_style, arg_dict['need_self_attn'], False, arg_dict['need_enc_attn'], True, arg_dict['tau'], arg_dict['greedy'],
					True, fm_lens, arg_dict['fm_recomp_lens'], arg_dict['less_len'])
				if arg_dict['dec_use_fm_iter']:
					with torch.set_grad_enabled(arg_dict['iter_sym']):
						fm_mask = (~(fm_padding_mask | (full_mask==constants.EOS_ID))) if arg_dict['fm_fill_eos'] else (~fm_padding_mask)
						fm_iter, fm_mask_iter = replace_toks(result['dec_fm']['hard_outputs'], result['dec_fm']['soft_outputs'], full_mask, 
							fm_mask, arg_dict['iter_keep_rate'], arg_dict['iter_random'])
						while (~fm_mask_iter).all():
							fm_iter, fm_mask_iter = replace_toks(result['dec_fm']['hard_outputs'], result['dec_fm']['soft_outputs'], full_mask, 
								fm_mask, arg_dict['iter_keep_rate'], arg_dict['iter_random'])
						fm_emb_iter = self.embed(fm_iter, longest_dec_pos_emb)
						result['fm_mask_iter'] = fm_mask_iter
						result['dec_fm_iter'] = self.decode(fm_emb_iter, to_style_dec_emb, longest_dec_pos_emb_full, None, enc_outputs, seq_mask_b, fm_padding_mask_b,
							fm_mask_iter, to_style, arg_dict['need_self_attn'], False, arg_dict['need_enc_attn'])
				if arg_dict['bt']:
					bt_emb = self.embed(result['dec_fm']['hard_outputs' if arg_dict['bt_sg'] else 'soft_outputs'], longest_enc_pos_emb, soft=not arg_dict['bt_sg'])
					if arg_dict['bt_use_recomp_lens'] and arg_dict['fm_recomp_lens']:
						fm_padding_mask_b, fm_lens = result['dec_fm']['hard_outputs_padding_mask_t_with_eos'], result['dec_fm']['hard_outputs_lens_with_eos']
					fm_enc_outputs, _ = self.encode(bt_emb, to_style_enc_emb, fm_padding_mask_b)
					dec_input_emb = para_x_dec_emb if arg_dict['dec_mask_mode'] else self.embed(uniform_copy(result['dec_fm']['hard_outputs'], fm_lens, lens, seq_mask), longest_dec_pos_emb)
					result['bt'] = self.decode(dec_input_emb, style_dec_emb, longest_dec_pos_emb_full, None, fm_enc_outputs, fm_padding_mask_b, seq_mask_b,
						mx_dec_mask, style)
			if arg_dict['dec_use_fc'] or arg_dict['fc_bt']:
				with torch.no_grad():
					dec_len = max((px.size(0) if arg_dict['fm_use_pseudo_len'] else x.size(0)) + arg_dict['extra_len'], 2)
					fm_lens = torch.clamp((plens if arg_dict['fm_use_pseudo_len'] else lens) + arg_dict['extra_len'], min=2)
					fm_padding_mask = get_padding_mask_on_size(dec_len, fm_lens)
					fm_padding_mask_b = fm_padding_mask.t().contiguous()
					if arg_dict['dec_mask_mode']:
						full_mask = x.new_full((dec_len, bsz), constants.MSK_ID).masked_fill_(fm_padding_mask, constants.PAD_ID)
						if arg_dict['fm_fill_eos']:
							full_mask.scatter_(0, fm_lens.unsqueeze(0)-1, constants.EOS_ID)
						fm_emb = self.embed(full_mask, longest_dec_pos_emb)
					else:
						fm_emb = self.embed(uniform_copy(x, lens, fm_lens, fm_padding_mask), longest_dec_pos_emb)
					
					mask = (fm_padding_mask | (full_mask==constants.EOS_ID)) if arg_dict['dec_mask_mode'] else fm_padding_mask
					fc_iter_result = self.decode(fm_emb, to_style_dec_emb, longest_dec_pos_emb_full, None, enc_outputs, seq_mask_b, fm_padding_mask_b, None, 
						to_style, require_outputs=True, tau=1, greedy=True, fill_eos=True, dec_lens=fm_lens)
					outputs = fc_iter_result['hard_outputs']
					scores = fc_iter_result['soft_outputs'].max(-1)[0].masked_fill_(mask, 1)
					logits = fc_iter_result['logits']
					fm_lens_no_eos = fm_lens - 1
					for i in range(1, arg_dict['fc_iter_num']):
						select_num = torch.clamp((fm_lens_no_eos.float()*(1.0-i/arg_dict['fc_iter_num'])).long(), min=1)
						# scores.masked_fill_(mask, scores.max()+1)
						worst_ids = select_worst(scores, select_num)
						outputs.view(-1)[worst_ids] = constants.MSK_ID
						# masked_outputs_list.append(outputs==constants.MSK_ID)
						fm_emb_iter = self.embed(outputs, longest_dec_pos_emb)
						fc_iter_result = self.decode(fm_emb_iter, to_style_dec_emb, longest_dec_pos_emb_full, None, enc_outputs, seq_mask_b, fm_padding_mask_b,
							None, to_style, require_outputs=True, tau=1, greedy=True, fill_eos=not arg_dict['fm_fill_eos'], dec_lens=fm_lens)
						# new_probs = decode_result_iter['soft_outputs'].max(-1)[0] if fm_fill_eos else decode_result_iter['soft_outputs'].gather(-1, decode_result_iter['hard_outputs'].unsqueeze(-1)).squeeze(-1)
						# new_outputs, new_scores = self.infer(emit_mi, ret_mi, mi_alpha, fm_emb_iter, to_style_dec_emb, style_dec_emb, longest_dec_pos_emb_full, 
						# 	enc_outputs, seq_mask_b, fm_padding_mask_b, to_style, style, fm_lens)
						scores.view(-1)[worst_ids] = fc_iter_result['soft_outputs'].max(-1)[0].view(-1)[worst_ids]
						outputs.view(-1)[worst_ids] = fc_iter_result['hard_outputs'].view(-1)[worst_ids]
						logits.view(-1, self.vocab_size)[worst_ids] = fc_iter_result['logits'].view(-1, self.vocab_size)[worst_ids]
				if arg_dict['dec_use_fc']:
					
					# assert (scores[mask] == 1).all().item()
					fc_input, fc_mask = mask_toks(arg_dict['fc_mask_mode'], outputs, scores, fm_padding_mask, fm_lens, arg_dict['fc_mask_rate'], arg_dict['fc_mask_largest'])
					while (~fc_mask).all():
						fc_input, fc_mask = mask_toks(arg_dict['fc_mask_mode'], outputs, scores, fm_padding_mask, fm_lens, arg_dict['fc_mask_rate'], arg_dict['fc_mask_largest'])
					
					# fc_input_emb = self.embed(fc_input, longest_dec_pos_emb)
					fc_input_tok_emb = self.token_embed(fc_input)
					fc_input_emb = self.add_tok_and_pos(fc_input_tok_emb, longest_dec_pos_emb)
					
					result['dec_fc'] = self.decode(fc_input_emb, to_style_dec_emb, longest_dec_pos_emb_full, None, enc_outputs, seq_mask_b, fm_padding_mask_b,
								None, to_style, require_outputs=arg_dict['fc_out'], tau=arg_dict['tau'], greedy=arg_dict['greedy'],
								need_enc_attn=arg_dict['ctr_fc_good_cov'])
					result['dec_fc']['fc_logits_target'] = logits
					result['dec_fc']['fc_hard_target'] = outputs
					result['dec_fc']['fc_mask'] = fc_mask
					result['dec_fc']['fc_lens'] = fm_lens
					result['dec_fc']['fc_padding_mask'] = fm_padding_mask
					result['dec_fc']['fc_padding_mask_b'] = fm_padding_mask_b
					with torch.no_grad():
						if arg_dict['ctr_fc_good_cov']:
							good_cov = coverage_score(result['dec_fc']['attn_weights'][-1]['enc_attn_weights'], lens, fm_lens,
								seq_mask_b, fm_padding_mask_b, arg_dict['cov_mode'], False, arg_dict['cov_with_start'])
						else:
							good_cov = None
					

					if arg_dict['fc_dctr']:
						result['fc_dctr'] = []
						result['fc_dctr_covs'] = []
						result['fc_dctr_word_ims'] = []
						result['fc_dctr_unchange'] = []
						result['fc_dctr_masks'] = []

						neg_samples = generate_negative_samples_by_delete(arg_dict['ctr_n'], fm_padding_mask, fc_mask, fm_lens, arg_dict['ctr_kmin'], arg_dict['ctr_kmax'], arg_dict['ctr_word_im'])
						for sample in neg_samples:
							ctr_padding_mask, ctr_mx_mask, ctr_positions, ctr_word_ims = sample
							ctr_lens = (~ctr_padding_mask).long().sum(0)
							pos_emb = self.dec_pos_emb_layer(device=device, detail_pos=ctr_positions)
							ctr_input_emb = self.add_tok_and_pos(fc_input_tok_emb, pos_emb)
							fc_dctr_dec_result = self.decode(ctr_input_emb, to_style_dec_emb, longest_dec_pos_emb_full, pos_emb, 
								enc_outputs, seq_mask_b, ctr_padding_mask.t(), None, to_style, need_enc_attn=arg_dict['ctr_fc_bad_cov'])
							result['fc_dctr'].append(fc_dctr_dec_result['logits'])
							result['fc_dctr_word_ims'].append(ctr_word_ims)
							with torch.no_grad():
								if arg_dict['ctr_fc_bad_cov']:
									bad_cov = coverage_score(fc_dctr_dec_result['attn_weights'][-1]['enc_attn_weights'], lens, ctr_lens,
																		seq_mask_b, ctr_padding_mask.t(), arg_dict['cov_mode'], False, arg_dict['cov_with_start'])
									result['fc_dctr_covs'].append(good_cov / bad_cov)
								else:
									result['fc_dctr_covs'].append(good_cov)
							result['fc_dctr_unchange'].append(ctr_lens==fm_lens)
							result['fc_dctr_masks'].append(ctr_mx_mask)
					if arg_dict['fc_ictr']:
						result['fc_ictr'] = []
						result['fc_ictr_word_ims'] = []
						result['fc_ictr_masks'] = []
						result['fc_ictr_filling_masks'] = []
						result['fc_ictr_covs'] = []
						neg_samples = generate_negative_samples_by_insert(arg_dict['ctr_n'], fc_input, fm_padding_mask, fm_lens, arg_dict['ctr_kmin'], arg_dict['ctr_kmax'], arg_dict['ctr_word_im'])
						for sample in neg_samples:
							ctr_input, ctr_padding_mask, ctr_mx_mask, ctr_word_ims, ctr_filling_mask = sample
							ctr_lens = (~ctr_padding_mask).long().sum(0)
							ctr_input_emb = self.embed(ctr_input, longest_dec_pos_emb)
							fc_ictr_dec_result = self.decode(ctr_input_emb, to_style_dec_emb, longest_dec_pos_emb_full, None, 
								enc_outputs, seq_mask_b, ctr_padding_mask.t(), None, to_style, need_enc_attn=arg_dict['ctr_fc_bad_cov'])
							result['fc_ictr'].append(fc_ictr_dec_result['logits'])
							result['fc_ictr_word_ims'].append(ctr_word_ims)
							result['fc_ictr_masks'].append(ctr_mx_mask)
							result['fc_ictr_filling_masks'].append(ctr_filling_mask)
							with torch.no_grad():
								if arg_dict['ctr_fc_bad_cov']:
									bad_cov = coverage_score(fc_ictr_dec_result['attn_weights'][-1]['enc_attn_weights'], lens, ctr_lens,
																		seq_mask_b, ctr_padding_mask.t(), arg_dict['cov_mode'], False, arg_dict['cov_with_start'])
									result['fc_ictr_covs'].append(good_cov / bad_cov)
								else:
									result['fc_ictr_covs'].append(good_cov)

				if arg_dict['fc_bt']:
					bt_emb = self.embed(outputs, longest_enc_pos_emb)
					fc_enc_outputs, _ = self.encode(bt_emb, to_style_enc_emb, fm_padding_mask_b)
					# dec_input_emb = para_x_dec_emb if arg_dict['dec_mask_mode'] else self.embed(uniform_copy(outputs, fm_lens, lens, seq_mask), longest_dec_pos_emb)
					if arg_dict['dec_mask_mode']:
						dec_input_tok = para_x_dec_tok
						dec_input_tok_emb = para_x_dec_tok_emb
						dec_input_emb = para_x_dec_emb
					else:
						dec_input_tok = uniform_copy(outputs, fm_lens, lens, seq_mask)
						dec_input_tok_emb = self.token_embed(dec_input_tok)
						dec_input_emb = self.add_tok_and_pos(dec_input_tok_emb, longest_dec_pos_emb)
					result['fc_bt'] = self.decode(dec_input_emb, style_dec_emb, longest_dec_pos_emb_full, None, fc_enc_outputs, fm_padding_mask_b, seq_mask_b,
						mx_dec_mask, style)
					if arg_dict['fc_bt_dctr']:
						result['fc_bt_dctr'] = []
						result['fc_bt_dctr_word_ims'] = []
						result['fc_bt_dctr_unchange'] = []
						result['fc_bt_dctr_masks'] = []
						neg_samples = generate_negative_samples_by_delete(arg_dict['ctr_n'], seq_mask, mx_dec_mask, lens, arg_dict['ctr_kmin'], arg_dict['ctr_kmax'], arg_dict['ctr_word_im'])
						for sample in neg_samples:
							ctr_padding_mask, ctr_mx_mask, ctr_positions, ctr_word_ims = sample
							ctr_lens = (~ctr_padding_mask).long().sum(0)
							pos_emb = self.dec_pos_emb_layer(device=device, detail_pos=ctr_positions)
							ctr_input_emb = self.add_tok_and_pos(dec_input_tok_emb, pos_emb)
							result['fc_bt_dctr'].append(self.decode(ctr_input_emb, style_dec_emb, longest_dec_pos_emb_full, pos_emb, 
								fc_enc_outputs, fm_padding_mask_b, ctr_padding_mask.t(), ctr_mx_mask, style)['logits'])
							result['fc_bt_dctr_word_ims'].append(ctr_word_ims)
							result['fc_bt_dctr_unchange'].append(ctr_lens==lens)
							result['fc_bt_dctr_masks'].append(ctr_padding_mask if ctr_mx_mask is None else ctr_mx_mask)
					if arg_dict['fc_bt_ictr']:
						result['fc_bt_ictr'] = []
						result['fc_bt_ictr_word_ims'] = []
						result['fc_bt_ictr_masks'] = []
						result['fc_bt_ictr_filling_masks'] = []
						neg_samples = generate_negative_samples_by_insert(arg_dict['ctr_n'], dec_input_tok, seq_mask, lens, arg_dict['ctr_kmin'], arg_dict['ctr_kmax'], arg_dict['ctr_word_im'])
						for sample in neg_samples:
							ctr_input, ctr_padding_mask, ctr_mx_mask, ctr_word_ims, ctr_filling_mask = sample
							ctr_input_emb = self.embed(ctr_input, longest_dec_pos_emb)
							result['fc_bt_ictr'].append(self.decode(ctr_input_emb, style_dec_emb, longest_dec_pos_emb_full, None, 
								fc_enc_outputs, fm_padding_mask_b, ctr_padding_mask.t(), ctr_mx_mask if mx_dec_mask is not None else None, style)['logits'])
							result['fc_bt_ictr_word_ims'].append(ctr_word_ims)
							result['fc_bt_ictr_masks'].append(ctr_padding_mask if mx_dec_mask is None else ctr_mx_mask)
							result['fc_bt_ictr_filling_masks'].append(ctr_filling_mask)


		return result

	def infer(self, emit_mi, ret_mi, mi_alpha, fm_emb, to_style_dec_emb, style_dec_emb, longest_dec_pos_emb_full, 
		enc_outputs, seq_mask_b, fm_padding_mask_b, to_style, style, fm_lens, need_enc_attn):
		result = self.decode(fm_emb, to_style_dec_emb, longest_dec_pos_emb_full, None, enc_outputs, seq_mask_b, fm_padding_mask_b, None, 
			to_style, need_enc_attn=need_enc_attn)
		logits_to_style = result['logits']
		enc_attn_weights = result['attn_weights'][-1]['enc_attn_weights'] if need_enc_attn else None
		lprobs_to_style = F.log_softmax(logits_to_style, -1)
		if emit_mi or ret_mi:
			logits_style = self.decode(fm_emb, style_dec_emb, longest_dec_pos_emb_full, None, enc_outputs, seq_mask_b, fm_padding_mask_b, None, 
				style)['logits']
			probs = (F.softmax(logits_style, -1) + F.softmax(logits_to_style, -1)) * 0.5
			mi_mix = lprobs_to_style - mi_alpha * torch.log(probs + 1e-10)
		emit_scores = mi_mix if emit_mi else lprobs_to_style
		outputs = emit_scores.argmax(-1)
		outputs.scatter_(0, (fm_lens-1).unsqueeze(0), constants.EOS_ID)
		ret_scores = mi_mix if ret_mi else lprobs_to_style
		ret_scores = ret_scores.gather(-1, outputs.unsqueeze(-1)).squeeze(-1)
		return outputs, ret_scores, enc_attn_weights

	def get_len_penalty(self, lens, fm_lens, simple_lp, lp_value, lp_rela, lp_cb_rela, lp_cb_add, lp_cb_simple, lp_cb_value):
		lens_for_lp = (fm_lens - lens).abs() + 1 if lp_rela else fm_lens
		len_penalty = (lens_for_lp.float() ** lp_value) if simple_lp else ((5.0 + lens_for_lp.float()) ** lp_value / (6.0 ** lp_value))
		if lp_cb_rela:
			rela_lens = (fm_lens - lens).abs() + 1
			cb_len_penalty = (rela_lens.float() ** lp_cb_value) if lp_cb_simple else ((5.0 + rela_lens.float()) ** lp_cb_value / (6.0 ** lp_cb_value))
			if lp_cb_add:
				len_penalty = len_penalty + cb_len_penalty
			else:
				len_penalty = len_penalty * cb_len_penalty
		return len_penalty

	def iterative_transfer(self, batch_dict, fm_fill_eos, len_offset_l, len_offset_r, dec_mask_mode, mask_repeat, simple_lp, lp_value, lp_rela,
		lp_cb_rela, lp_cb_add, lp_cb_simple, lp_cb_value, iter_num, emit_mi, ret_mi, mi_alpha, all_iter_eval, rescore_mode, style_tool, beta, at_model,
		add_cov, cov_mode, cov_weight, cov_inv, cov_with_start):
		style, x, lens, seq_mask_b = [batch_dict[name] for name in ['style', 'x', 'lens', 'seq_mask_b']]
		style_emb = self.style_emb_layer(style)
		to_style = self.get_target_style(style)
		org_x = x
		org_to_style = to_style
		org_style = style
		org_seq_mask_b = seq_mask_b
		org_lens = lens
		to_style_emb = self.style_emb_layer(to_style)
		bsz = style.size(0)
		# need an extra position for the style
		max_positions = x.size(0) + len_offset_r + 1
		longest_enc_pos_emb = self.position_embed(max_positions, style.device, True).expand(max_positions, bsz, self.emb_size).contiguous()
		longest_dec_pos_emb = longest_enc_pos_emb if self.share_pos_emb else self.position_embed(max_positions, style.device, False).expand(max_positions, bsz, self.emb_size).contiguous()
		longest_dec_pos_emb_full = longest_dec_pos_emb
		style_enc_pos_emb, style_dec_pos_emb, longest_enc_pos_emb, longest_dec_pos_emb = longest_enc_pos_emb[0], longest_dec_pos_emb[0], longest_enc_pos_emb[1:], longest_dec_pos_emb[1:]
		style_enc_emb = style_emb + style_enc_pos_emb
		to_style_dec_emb = to_style_emb + style_dec_pos_emb
		style_dec_emb = style_emb + style_dec_pos_emb
		
		enc_input_emb = self.embed(x, longest_enc_pos_emb)
		enc_outputs, _ = self.encode(enc_input_emb, style_enc_emb, seq_mask_b)
		beam_size = len_offset_l + len_offset_r + 1
		x, longest_dec_pos_emb_full, enc_outputs = repeat_tensors(beam_size, 1, (x, longest_dec_pos_emb_full, enc_outputs))
		lens, style_dec_emb, to_style_dec_emb, style, to_style, seq_mask_b = repeat_tensors(beam_size, 0, (lens, style_dec_emb, to_style_dec_emb, style, to_style, seq_mask_b))
		longest_dec_pos_emb = longest_dec_pos_emb_full[1:]
		
		dec_len = x.size(0) + len_offset_r
		fm_lens = lens.view(bsz, beam_size) + torch.arange(-len_offset_l, len_offset_r+1, dtype=torch.long, device=style.device).unsqueeze(0)
		fm_lens = torch.clamp(fm_lens.view(-1), min=2)
		fm_padding_mask = get_padding_mask_on_size(dec_len, fm_lens)
		fm_padding_mask_b = fm_padding_mask.t().contiguous()
		if dec_mask_mode:
			full_mask = x.new_full((dec_len, bsz*beam_size), constants.MSK_ID).masked_fill_(fm_padding_mask, constants.PAD_ID)
			if fm_fill_eos:
				full_mask.scatter_(0, fm_lens.unsqueeze(0)-1, constants.EOS_ID)
			fm_emb = self.embed(full_mask, longest_dec_pos_emb)
		else:
			fm_emb = self.embed(uniform_copy(x, lens, fm_lens, fm_padding_mask), longest_dec_pos_emb)
		
		mask = (fm_padding_mask | (full_mask==constants.EOS_ID)) if dec_mask_mode else fm_padding_mask
		# decode_result = self.decode(fm_emb, to_style_dec_emb, longest_dec_pos_emb_full, enc_outputs, seq_mask_b, fm_padding_mask_b, None, 
		# 	to_style, require_outputs=True, tau=1, greedy=True, fill_eos=True, dec_lens=fm_lens)
		# outputs = decode_result['hard_outputs']
		# scores = decode_result['soft_outputs'].gather(-1, outputs.unsqueeze(-1)).squeeze(-1).masked_fill_(mask, 1)
		outputs, scores, enc_attn_weights = self.infer(emit_mi, ret_mi, mi_alpha, fm_emb, to_style_dec_emb, style_dec_emb, longest_dec_pos_emb_full, 
			enc_outputs, seq_mask_b, fm_padding_mask_b, to_style, style, fm_lens, add_cov)
		
		if iter_num > 1:
			outputs_list = [outputs.clone()]
			scores_list = [scores.clone()]
			attn_list = [enc_attn_weights]
			masked_outputs_list = []
			fm_lens_no_eos = fm_lens - 1
			for i in range(1, iter_num):
				select_num = torch.clamp((fm_lens_no_eos.float()*(1.0-i/iter_num)).long(), min=1)
				scores.masked_fill_(mask, scores.max()+1)
				worst_ids = select_worst(scores, select_num)
				if mask_repeat:
					repeat_mask = get_repeat_mask(outputs, mask)
					outputs[repeat_mask] = constants.MSK_ID
				outputs.view(-1)[worst_ids] = constants.MSK_ID
				masked_outputs_list.append(outputs==constants.MSK_ID)
				fm_emb_iter = self.embed(outputs, longest_dec_pos_emb)
				# decode_result_iter = self.decode(fm_emb_iter, to_style_dec_emb, longest_dec_pos_emb_full, enc_outputs, seq_mask_b, fm_padding_mask_b,
					# None, to_style, require_outputs=True, tau=1, greedy=True, fill_eos=not fm_fill_eos, dec_lens=fm_lens)
				# new_probs = decode_result_iter['soft_outputs'].max(-1)[0] if fm_fill_eos else decode_result_iter['soft_outputs'].gather(-1, decode_result_iter['hard_outputs'].unsqueeze(-1)).squeeze(-1)
				new_outputs, new_scores, enc_attn_weights = self.infer(emit_mi, ret_mi, mi_alpha, fm_emb_iter, to_style_dec_emb, style_dec_emb, longest_dec_pos_emb_full, 
					enc_outputs, seq_mask_b, fm_padding_mask_b, to_style, style, fm_lens, add_cov)
				scores.view(-1)[worst_ids] = new_scores.view(-1)[worst_ids]
				outputs.view(-1)[worst_ids] = new_outputs.view(-1)[worst_ids]
				if mask_repeat:
					scores[repeat_mask] = new_scores[repeat_mask]
					outputs[repeat_mask] = new_outputs[repeat_mask]
				outputs_list.append(outputs.clone())
				scores_list.append(scores.clone())
				attn_list.append(enc_attn_weights)
			# assert (scores[mask] == 1).all().item()
		if all_iter_eval and iter_num > 1:
			outputs = merge_iters(outputs_list, beam_size)
			scores = merge_iters(scores_list, beam_size)
			if add_cov:
				enc_attn_weights = merge_attn_iters(attn_list, beam_size)
			mask = repeat_tensors(iter_num, 2, mask.view(dec_len, bsz, beam_size)).view(dec_len, -1)
			fm_lens = repeat_tensors(iter_num, 1, fm_lens.view(bsz, beam_size)).view(-1)
			fm_padding_mask = repeat_tensors(iter_num, 2, fm_padding_mask.view(dec_len, bsz, beam_size)).view(dec_len, -1)
			fm_padding_mask_b = fm_padding_mask.t().contiguous()
			seq_mask_b = repeat_tensors(iter_num, 1, seq_mask_b.view(bsz, beam_size, -1)).view(bsz*beam_size*iter_num, -1)
			lens = repeat_tensors(iter_num, 1, lens.view(bsz, beam_size)).view(-1)
		
		if rescore_mode == 'reward':
			final_scores = reward(style_tool, org_to_style, org_x, org_lens, outputs, fm_lens, fm_padding_mask, fm_padding_mask_b, beta)
		else:
			if rescore_mode == 'at':
				final_scores = at_as_eval(at_model, org_x, org_style, org_to_style, org_seq_mask_b, outputs, fm_padding_mask_b)
			else:
				final_scores = scores

			final_scores = final_scores.masked_fill(mask, 0).sum(0)
			len_penalty = self.get_len_penalty(lens-1 if fm_fill_eos else lens, fm_lens-1 if fm_fill_eos else fm_lens, simple_lp, lp_value, lp_rela, lp_cb_rela, lp_cb_add, lp_cb_simple, lp_cb_value)
			final_scores = final_scores/len_penalty
		if add_cov:
			attn_scores = coverage_score(enc_attn_weights, lens, fm_lens, seq_mask_b, fm_padding_mask_b, cov_mode, cov_inv, cov_with_start)
			final_scores = final_scores + cov_weight * attn_scores


		best_ind = final_scores.view(bsz, -1).argmax(1, keepdim=True)
		result = {}
		result['to_style'] = org_to_style
		result['hard_outputs_lens_with_eos'] = fm_lens.view(bsz, -1).gather(-1, best_ind).squeeze(-1)
		best_ind_expand = best_ind.unsqueeze(0).expand(dec_len, bsz, 1)
		result['hard_outputs_padding_mask_with_eos'] = fm_padding_mask.view(dec_len, bsz, -1).gather(-1, best_ind_expand).squeeze(-1)
		result['hard_outputs_padding_mask_t_with_eos'] = result['hard_outputs_padding_mask_with_eos'].t().contiguous()
		# result['hard_outputs_padding_mask_with_eos'] = result['hard_outputs_padding_mask_t_with_eos'].t().contiguous()
		result['hard_outputs'] = outputs.view(dec_len, bsz, -1).gather(-1, best_ind_expand).squeeze(-1)
		scores = scores.exp()
		if all_iter_eval:
			if iter_num > 1:
				masked_outputs = merge_iters(masked_outputs_list, beam_size)
			for k in range(beam_size):
				for i in range(iter_num):
					result[f'outputs_offset_{k - len_offset_l}_iter_{i}'] = outputs.view(dec_len, bsz, beam_size, iter_num)[:, :, k, i]
					result[f'scores_offset_{k - len_offset_l}_iter_{i}'] = scores.view(dec_len, bsz, beam_size, iter_num)[:, :, k, i]
					result[f'final_scores_offset_{k - len_offset_l}_iter_{i}'] = final_scores.view(bsz, beam_size, iter_num)[:, k, i]
					result[f'lens_offset_{k - len_offset_l}_iter_{i}'] = fm_lens.view(bsz, beam_size, iter_num)[:, k, i]
					result[f'padding_offset_{k - len_offset_l}_iter_{i}'] = fm_padding_mask.view(dec_len, bsz, beam_size, iter_num)[:, :, k, i]
					result[f'padding_b_offset_{k - len_offset_l}_iter_{i}'] = result[f'padding_offset_{k - len_offset_l}_iter_{i}'].t().contiguous()
					if i < iter_num-1:
						result[f'masked_outputs_offset_{k - len_offset_l}_iter_{i}'] = masked_outputs.view(dec_len, bsz, beam_size, iter_num-1)[:, :, k, i]
			result['best_offset'] = best_ind // iter_num
			result['best_iter'] = best_ind % iter_num
		else:
			for i in range(iter_num-1):
				result[f'outputs_iter_{i}'] = outputs_list[i].view(dec_len, bsz, beam_size).gather(-1, best_ind_expand).squeeze(-1)
				result[f'masked_outputs_iter_{i}'] = masked_outputs_list[i].view(dec_len, bsz, beam_size).gather(-1, best_ind_expand).squeeze(-1)
				result[f'scores_iter_{i}'] = scores_list[i].view(dec_len, bsz, beam_size).gather(-1, best_ind_expand).squeeze(-1)
			result[f'scores_iter_{iter_num-1}'] = scores.view(dec_len, bsz, beam_size).gather(-1, best_ind_expand).squeeze(-1)
		return result

def mask_toks(mask_mode, x, scores, padding_mask, lens, mask_rate, largest):
	if mask_rate is None:
		mask_rate = random.uniform(0.3, 1.0)
	if mask_mode == 'random':
		return mask_random_toks(x, padding_mask, mask_rate)
	else:
		return mask_topk_toks(x, scores, padding_mask, lens, mask_rate, largest)
def mask_random_toks(x, padding_mask, mask_rate):
	random_mask = padding_mask.new_empty(padding_mask.size()).bernoulli_(mask_rate)
	random_mask = (~padding_mask) & random_mask
	new_x = x.masked_fill(random_mask, constants.MSK_ID)
	return new_x, random_mask

def mask_topk_toks(x, scores, padding_mask, lens, mask_rate, largest):
	bsz = x.size(1)
	if largest:
		scores = scores.masked_fill(padding_mask, 0)
	mask_num = torch.clamp((lens.float() * mask_rate).long(), min=1)
	max_mask_num = mask_num.max().item()
	top_ids = scores.topk(max_mask_num, 0, largest=largest)[1]
	top_mask = get_padding_mask_on_size(max_mask_num, mask_num, flip=True)
	offsets = torch.arange(0, bsz, dtype=torch.long, device=x.device)
	top_ids = top_ids * bsz + offsets.unsqueeze(0)
	top_ids = top_ids[top_mask]
	new_x = x.view(-1).scatter(0, top_ids, constants.MSK_ID).view(x.size())
	topk_mask = new_x == constants.MSK_ID
	return new_x, topk_mask


def replace_random_toks(hard_outputs, org_inputs, org_mx_mask, keep_rate):
	random_mask = org_mx_mask.new_empty(org_mx_mask.size()).bernoulli_(keep_rate)
	change_mask = (~random_mask) & org_mx_mask
	mx_mask = random_mask & org_mx_mask
	new_inputs = org_inputs.masked_scatter(change_mask, hard_outputs[change_mask])
	return new_inputs, mx_mask
def replace_best_toks(hard_outputs, soft_outputs, org_inputs, org_mx_mask, change_rate):
	seq_len, bsz = hard_outputs.size()
	probs = soft_outputs.max(-1)[0]
	probs = probs.masked_fill(~org_mx_mask, 0)
	mx_num = org_mx_mask.float().sum(0)
	replace_num = (mx_num * change_rate).long()
	max_replace_num = replace_num.max().item()
	top_ids = probs.topk(max_replace_num, 0, largest=True, sorted=True)[1]
	top_mask = get_padding_mask_on_size(max_replace_num, replace_num, flip=True)
	offsets = torch.arange(0, bsz, dtype=torch.long, device=hard_outputs.device)
	top_ids = top_ids * bsz + offsets.unsqueeze(0)
	top_ids = top_ids[top_mask]
	new_inputs = org_inputs.view(-1).scatter(0, top_ids, hard_outputs.view(-1)[top_ids]).view(seq_len, bsz)
	mx_mask = new_inputs == constants.MSK_ID
	return new_inputs, mx_mask

def replace_toks(hard_outputs, soft_outputs, org_inputs, org_mx_mask, keep_rate, random_mode):
	if random_mode:
		return replace_random_toks(hard_outputs, org_inputs, org_mx_mask, keep_rate)
	else:
		return replace_best_toks(hard_outputs, soft_outputs, org_inputs, org_mx_mask, 1-keep_rate)

def select_worst(probs, select_num):
	seq_len, bsz = probs.size()
	max_select_num = select_num.max().item()
	worst_ids = probs.topk(max_select_num, 0, largest=False, sorted=True)[1]
	worst_mask = get_padding_mask_on_size(max_select_num, select_num, flip=True)
	offsets = torch.arange(0, bsz, dtype=torch.long, device=probs.device)
	worst_ids = worst_ids * bsz + offsets.unsqueeze(0)
	worst_ids = worst_ids[worst_mask]
	# next_input = prev_outs.view(-1).scatter(0, worst_ids, constants.MSK_ID).view(seq_len, bsz)
	return worst_ids



def mask_input(x, keep_eos):
	mask = x!=constants.PAD_ID
	if keep_eos:
		mask = mask & (x!=constants.EOS_ID)
	return x.masked_fill(mask, constants.MSK_ID)

def uniform_copy(org_inputs, src_lens, tgt_lens, tgt_padding_mask):
	max_len = tgt_padding_mask.size(0)
	inds = torch.arange(0, max_len, dtype=torch.float, device=org_inputs.device)
	one_tok_mask = tgt_lens == 1
	if one_tok_mask.any():
		tgt_lens = tgt_lens.masked_fill(one_tok_mask, 2)
	inds = torch.round(inds.unsqueeze(-1) / (tgt_lens-1).unsqueeze(0).float() * (src_lens-1).unsqueeze(0).float()).long()
	inds = inds.masked_fill(tgt_padding_mask, 0)
	new_inputs = org_inputs.gather(0, inds).masked_fill(tgt_padding_mask, constants.PAD_ID)
	return new_inputs

def uniform_copy_by_pos(org_inputs, src_lens, tgt_lens, tgt_padding_mask, positions):
	max_len = tgt_padding_mask.size(0)
	inds = positions - 1
	one_tok_mask = tgt_lens == 1
	if one_tok_mask.any():
		tgt_lens = tgt_lens.masked_fill(one_tok_mask, 2)
	inds = torch.round(inds.float() / (tgt_lens-1).unsqueeze(0).float() * (src_lens-1).unsqueeze(0).float()).long()
	inds = inds.masked_fill(tgt_padding_mask, 0)
	new_inputs = org_inputs.gather(0, inds).masked_fill(tgt_padding_mask, constants.PAD_ID)
	return new_inputs


def get_repeat_mask(x, padding_mask):
	bsz = x.size(1)
	shift_x = torch.cat([x, x.new_full((1, bsz), constants.PAD_ID)], 0)
	first_mask = x == shift_x[1:]
	second_mask = torch.cat([first_mask.new_zeros((1, bsz)), first_mask[:-1]], 0)
	mask = first_mask | second_mask
	mask[padding_mask] = 0
	return mask

def merge_iters(outputs_list, beam_size):
	seq_len = outputs_list[0].size(0)
	bsz = outputs_list[0].size(1)//beam_size
	num_iters = len(outputs_list)
	return torch.cat([x.view(seq_len, bsz, beam_size, 1) for x in outputs_list], -1).view(seq_len, bsz * beam_size * num_iters)

def merge_attn_iters(attn_list, beam_size):
	qlen, klen = attn_list[0].size(1), attn_list[0].size(2)
	bsz = attn_list[0].size(0)//beam_size
	num_iters = len(attn_list)
	return torch.cat([x.view(bsz, beam_size, 1, qlen, klen) for x in attn_list], 2).view(bsz * beam_size * num_iters, qlen, klen)

def generate_negative_samples_by_delete(n, padding_mask, mx_mask, lens, k_min, k_max, need_word_im):
	samples = []
	weights = (~padding_mask).t().contiguous()
	weights = weights.scatter(1, lens.unsqueeze(-1)-1, 0).float()
	for i in range(n):
		k = random.randint(k_min, k_max)
		samples.append(remove_toks(weights, padding_mask, mx_mask, lens, k, need_word_im))
	return samples

def remove_toks(weights, padding_mask, mx_mask, lens, k, need_word_im):
	# max_remove_num = torch.clamp(lens-2, min=0)
	# weights = ~padding_mask.t().contiguous()
	# weights = weights.scatter(1, lens.unsqueeze(-1)-1, 0).float()
	device = lens.device
	bsz = lens.size(0)
	remove_num = torch.clamp(lens-2, min=0, max=k)
	max_num = remove_num.max().item()
	if max_num > 0:
		remove_mask = get_padding_mask_on_size(max_num, remove_num, 0, 1, flip=True)
		
		id_list_with_offset = []
		id_list = []

		for i in range(bsz):
			if remove_num[i] > 0:
				ids = torch.multinomial(weights[i], remove_num[i])
				if mx_mask is not None:
					id_mask = padding_mask.new_zeros(padding_mask.size(0))
					id_mask[ids] = 1
					while ((id_mask & mx_mask[:, i]) == mx_mask[:, i]).all():
						ids = torch.multinomial(weights[i], remove_num[i])
						id_mask = padding_mask.new_zeros(padding_mask.size(0))
						id_mask[ids] = 1
				ids_offset = ids * bsz + i
				id_list.append(ids)
				id_list_with_offset.append(ids_offset)
		# print(lens.max().item())
		# print(remove_num.sum().item())
		ids_with_offset = torch.cat(id_list_with_offset, 0)
		ids_no_offset = torch.cat(id_list, 0)
		new_padding_mask = padding_mask.view(-1).scatter(0, ids_with_offset, 1).view(padding_mask.size())
		if need_word_im:
			select_ids = ids_no_offset.new_full(remove_mask.size(), 1000)
			select_ids[remove_mask] = ids_no_offset
			select_ids.t_()
			orders = torch.arange(0, padding_mask.size(0), dtype=torch.long, device=device).unsqueeze(-1).unsqueeze(-1)
			diff = torch.abs(orders - select_ids.unsqueeze(0))
			diff = diff.min(1)[0]
			diff = diff.float() / lens.float()
			word_ims = 1 / (diff**2 + 1)
			# word_ims = word_ims.masked_fill(padding_mask, 0)

		else:
			word_ims = None
	else:
		new_padding_mask = padding_mask
		word_ims = torch.ones(padding_mask.size(), dtype=torch.float, device=device)

	positions = (~new_padding_mask).long().cumsum(0)
	if mx_mask is not None:
		new_mx_mask = mx_mask * (~new_padding_mask)
	else:
		new_mx_mask = None

	return new_padding_mask, new_mx_mask, positions, word_ims

def generate_negative_samples_by_insert(n, org_input, padding_mask, lens, k_min, k_max, need_word_im):
	samples = []
	weights = (~padding_mask).t().contiguous().float()
	org_input_t = org_input.t().contiguous()
	for i in range(n):
		k = random.randint(k_min, k_max)
		samples.append(insert_toks(org_input_t, weights, padding_mask, lens, k, need_word_im))
	return samples
def insert_toks(org_input_t, weights, padding_mask, lens, k, need_word_im):
	bsz = lens.size(0)
	device = lens.device
	new_padding_mask = torch.cat([padding_mask.new_zeros((k, bsz)), padding_mask], 0)
	new_seq_len = new_padding_mask.size(0)

	ids = torch.multinomial(weights, 1)
	get_shifts = torch.arange(k-1, -1, -1, dtype=torch.long, device=device).unsqueeze(0)
	fill_shifts = torch.arange(1, k+1, dtype=torch.long, device=device).unsqueeze(0)
	get_ids = (ids - get_shifts).clamp(min=0)
	fill_ids = (ids + fill_shifts)
	toks = org_input_t.gather(1, get_ids)


	new_input = org_input_t.new_empty((bsz, new_seq_len))
	filling_mask = padding_mask.new_zeros((bsz, new_seq_len))
	filling_mask.scatter_(1, fill_ids, 1)
	new_input[filling_mask] = toks.view(-1)
	new_input[~filling_mask] = org_input_t.view(-1)
	new_input.t_()

	new_mx_mask = (new_input == constants.MSK_ID)
	if need_word_im:
		orders = torch.arange(0, new_seq_len, dtype=torch.long, device=device).unsqueeze(-1).unsqueeze(-1)
		diff = torch.abs(orders - fill_ids.unsqueeze(0))
		diff = diff.min(2)[0]
		diff = diff.float() / (lens.float() + k)
		word_ims = 1 / (diff**2 + 1)
	else:
		word_ims = None
	return new_input, new_padding_mask, new_mx_mask, word_ims, filling_mask


