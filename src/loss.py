import math
import subprocess
import re
# import numpy as np
# from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import check_values
from layers import cat_zeros_at_start
# from layers import repeat_tensors


	
def seq_ce_logits_loss(y, t, t_lens, mask, size_average, batch_mask=None, smooth=0, reduction=True):
	# check_values(y, 'logits', False)
	y = F.log_softmax(y, 2)

	if t.dim() == 2:
		loss = -torch.gather(y, 2, t.unsqueeze(-1)).squeeze(-1)
		if smooth > 0:
			smooth_loss = -y.sum(-1)
			eps = smooth / y.size(-1)
			loss = (1-smooth)*loss + eps*smooth_loss
	else:
		loss = y * t
		loss = - loss.sum(2)
	
	loss = loss.masked_fill(mask, 0)
	loss = loss.sum(0) / t_lens.masked_fill(t_lens==0, 1).float()
	
	if batch_mask is not None:
		loss = loss.masked_fill(batch_mask, 0)

	return (loss.mean() if size_average else loss.sum()) if reduction else loss

def reform_target(t, filling_mask):
	t_reform = t.new_zeros(filling_mask.size())
	t_reform[~filling_mask] = t.t().contiguous().view(-1)
	t_reform.t_()
	return t_reform

def reform_pm(new_padding_mask, filling_mask):
	padding_mask_reform = new_padding_mask.masked_fill(filling_mask.t(), 1)
	return padding_mask_reform

def reform_mm(new_mx_mask, filling_mask):
	mx_mask_reform = new_mx_mask.masked_fill(filling_mask.t(), 0)
	return mx_mask_reform

def shrink(x, filling_mask):
	bsz = x.size(1)
	x = x.t()[~filling_mask]
	x = x.view(bsz, -1).t().contiguous()
	return x

def shrink_3d(x, filling_mask):
	bsz = x.size(1)
	fsize = x.size(-1)
	x = x.transpose(0, 1).masked_select(~filling_mask.unsqueeze(-1))
	x = x.view(bsz, -1, fsize).transpose(0, 1).contiguous()
	return x

def seq_contrast_loss(y1, y2_list, mask1, mask2_list, t, use_t1, use_t2, margin, by_seq, weights_list, batch_weights_list, unchange_masks, size_average):
	y1 = F.log_softmax(y1, 2)
	if use_t1:
		y1 = torch.gather(y1, 2, t.unsqueeze(-1)).squeeze(-1)
	else:
		y1 = y1.max(2)[0]
	if by_seq:
		y1 = y1.masked_fill(mask1, 0)
		lens1 = (~mask1).float().sum(0)
		y1 = y1.sum(0) / lens1
	loss = 0
	
	for y2, mask2, weights, batch_weights, unchange_mask in zip(y2_list, mask2_list, weights_list, batch_weights_list, unchange_masks):

		lens2 = (~mask2).float().sum(0)
		
		y2 = F.log_softmax(y2, 2)
		if use_t2:
			y2 = torch.gather(y2, 2, t.unsqueeze(-1)).squeeze(-1)
		else:
			y2 = y2.max(2)[0]
		
		if by_seq:
			y2 = y2.masked_fill(mask2, 0)
			y2 = y2.sum(0) / lens2
			cur_loss = torch.clamp(margin - y1 + y2, min=0)
			# loss = loss + torch.clamp(margin - y1 + y2, min=0).masked_fill(unchange_mask, 0)
		else:
			cur_loss = torch.clamp(margin - y1 + y2, min=0)
			if weights is None:
				cur_loss = cur_loss.masked_fill(mask2, 0).sum(0) / lens2
			else:
				weights = weights.masked_fill(mask2, 0)
				cur_loss = (cur_loss * weights).sum(0) / weights.sum(0)
			
		if batch_weights is not None:
			cur_loss = cur_loss * batch_weights
		loss = loss + cur_loss.masked_fill(unchange_mask, 0)

	return loss.mean() if size_average else loss.sum()

def seq_contrast_insert_loss(y1, y2_list, mask1, mask2_list, t, use_t1, margin, by_seq, weights_list, batch_weights_list, filling_masks, size_average):
	y1 = F.log_softmax(y1, 2)
	if use_t1:
		y1 = torch.gather(y1, 2, t.unsqueeze(-1)).squeeze(-1)
	else:
		y1 = y1.max(2)[0]
	lens1 = (~mask1).float().sum(0)
	if by_seq:
		y1 = y1.masked_fill(mask1, 0)
		y1 = y1.sum(0) / lens1
	loss = 0
	
	for y2, mask2, weights, batch_weights, filling_mask in zip(y2_list, mask2_list, weights_list, batch_weights_list, filling_masks):

		
		
		y2 = F.log_softmax(y2, 2)	
		y2 = y2.max(2)[0]
		
		if by_seq:
			lens2 = (~mask2).float().sum(0)
			y2 = y2.masked_fill(mask2, 0)
			y2 = y2.sum(0) / lens2
			cur_loss = torch.clamp(margin - y1 + y2, min=0)
			# loss = loss + torch.clamp(margin - y1 + y2, min=0).masked_fill(unchange_mask, 0)
		else:
			y2 = shrink(y2, filling_mask)
			cur_loss = torch.clamp(margin - y1 + y2, min=0)
			if weights is None:
				cur_loss = cur_loss.masked_fill(mask1, 0).sum(0) / lens1
			else:
				weights = shrink(weights, filling_mask).masked_fill(mask1, 0)
				cur_loss = (cur_loss * weights).sum(0) / weights.sum(0)
			
		if batch_weights is not None:
			cur_loss = cur_loss * batch_weights
		loss = loss + cur_loss

	return loss.mean() if size_average else loss.sum()
def mx_contrast_loss(y1, y2_list, mask1, mask2_list, t, use_t1, use_t2, margin, by_seq, weights_list, batch_weights_list, unchange_masks, size_average):
	y1 = F.log_softmax(y1, -1)
	if use_t1:
		y1 = torch.gather(y1, 1, t[mask1].unsqueeze(-1)).squeeze(-1)
	else:
		y1 = y1.max(-1)[0]
	if by_seq:
		y1_full = mask1.new_zeros(mask1.size(), dtype=torch.float).masked_scatter(mask1, y1)
		mask1_num = mask1.float().sum(0)
		y1_full = y1_full.sum(0) / mask1_num
	loss = 0

	for y2, mask2, weights, batch_weights, unchange_mask in zip(y2_list, mask2_list, weights_list, batch_weights_list, unchange_masks):
		
		y2 = F.log_softmax(y2, -1)
		if use_t2:
			y2 = torch.gather(y2, 1, t[mask2].unsqueeze(-1)).squeeze(-1)
		else:
			y2 = y2.max(-1)[0]

		if by_seq:
			y2_full = mask2.new_zeros(mask2.size(), dtype=torch.float).masked_scatter(mask2, y2)
			mask2_num = mask2.float().sum(0)
			y2_full = y2_full.sum(0) / mask2_num
			cur_loss = torch.clamp(margin - y1_full + y2_full, min=0).masked_fill(unchange_mask, 0)
			if batch_weights is not None:
				cur_loss = cur_loss * batch_weights
			loss = loss + cur_loss.sum()
		else:
			y1_view = y1[mask2[mask1]]
			if weights is None:
				mask2_num = mask2.float().sum(0, keepdim=True).expand_as(mask2)[mask2]
				cur_loss = torch.clamp(margin - y1_view + y2, min=0) / mask2_num
			else:
				weights = weights.masked_fill(~mask2, 0)
				weights_view = (weights / weights.sum(0))[mask2]
				cur_loss = torch.clamp(margin - y1_view + y2, min=0) * weights_view
			cur_loss = cur_loss.masked_fill(unchange_mask.unsqueeze(0).expand_as(mask2)[mask2], 0)
			if batch_weights is not None:
				batch_weights_view = batch_weights.unsqueeze(0).expand_as(mask2)[mask2]
				cur_loss = cur_loss * batch_weights_view
			loss = loss + cur_loss.sum()

	if size_average:
		loss = loss / mask1.size(1)

	return loss



def mx_contrast_insert_loss(y1, y2_list, mask1, mask2_list, t, use_t1, margin, by_seq, weights_list, batch_weights_list, filling_masks, size_average):
	y1 = F.log_softmax(y1, -1)
	if use_t1:
		y1 = torch.gather(y1, 1, t[mask1].unsqueeze(-1)).squeeze(-1)
	else:
		y1 = y1.max(-1)[0]
	if by_seq:
		y1_full = mask1.new_zeros(mask1.size(), dtype=torch.float).masked_scatter(mask1, y1)
		mask1_num = mask1.float().sum(0)
		y1_full = y1_full.sum(0) / mask1_num
	loss = 0
	

	for y2, mask2, weights, batch_weights, filling_mask in zip(y2_list, mask2_list, weights_list, batch_weights_list, filling_masks):
		
		y2 = F.log_softmax(y2, -1)
		y2 = y2.max(-1)[0]
		
		y2_full = mask2.new_zeros(mask2.size(), dtype=torch.float).masked_scatter(mask2, y2)
		if by_seq:
			
			mask2_num = mask2.float().sum(0)
			y2_full = y2_full.sum(0) / mask2_num
			cur_loss = torch.clamp(margin - y1_full + y2_full, min=0)
			if batch_weights is not None:
				cur_loss = cur_loss * batch_weights
			loss = loss + cur_loss.sum()
		else:
			y2_view = shrink(y2_full, filling_mask)[mask1]
			if weights is None:
				mask1_num = mask1.float().sum(0, keepdim=True).expand_as(mask1)[mask1]
				cur_loss = torch.clamp(margin - y1 + y2_view, min=0) / mask1_num
			else:
				weights = shrink(weights, filling_mask).masked_fill(~mask1, 0)
				weights_view = (weights / weights.sum(0))[mask1]
				cur_loss = torch.clamp(margin - y1 + y2_view, min=0) * weights_view
			if batch_weights is not None:
				batch_weights_view = batch_weights.unsqueeze(0).expand_as(mask1)[mask1]
				cur_loss = cur_loss * batch_weights_view
			loss = loss + cur_loss.sum()

	if size_average:
		loss = loss / mask1.size(1)

	return loss


def seq_ce_logits_topk_loss(y, topi, topp, t_lens, mask, size_average, smooth=0):
	# check_values(y, 'logits', False)
	y = F.log_softmax(y, 2)

	loss = torch.gather(y, 2, topi) * topp
	loss = -loss.sum(-1)
	if smooth > 0:
		smooth_loss = -y.sum(-1)
		eps = smooth / y.size(-1)
		loss = (1-smooth)*loss + eps*smooth_loss
	

	loss = loss.masked_fill(mask, 0)
	loss = loss.sum(0) / t_lens.masked_fill(t_lens==0, 1).float()
	
	

	return loss.mean() if size_average else loss.sum()


def seq_acc(y, t, t_lens, mask, size_average):
	with torch.no_grad():
		# y = y.detach()
		pred = torch.argmax(y, 2)
		loss = (pred == t) & (~mask)
		loss = loss.float().sum(0) / t_lens.float()

	return loss.mean().item() if size_average else loss.sum().item()

def unit_acc(y, t, size_average = True, reduction=True):
	with torch.no_grad():
		# y = y.detach()
		pred = torch.argmax(y, 1)
		loss = (pred == t).float()

	return (loss.mean().item() if size_average else loss.sum().item()) if reduction else loss

def adv_loss(y, from_style, to_style, mode, size_average):
	if mode == 'ac':
		y = F.log_softmax(y, 1)
		loss = - y[:, -1]
	elif mode == 'ent':
		logp = F.log_softmax(y, 1)
		p = F.softmax(y, 1)
		loss = torch.sum(p * logp, 1)
	elif mode == 'uni':
		logp = F.log_softmax(y, 1)
		t = y.new_full(y.size(), 1.0/y.size(1))
		loss = - torch.sum(t * logp, 1)
	elif mode == 'src':
		# if from_style.size(0) != y.size(0):
			# from_style = repeat_tensors(y.size(0)//from_style.size(0), 0, from_style)
		loss = F.cross_entropy(y, from_style, reduction='none')
	elif mode == 'mtsf':
		loss = - F.cross_entropy(y, to_style, reduction='none')
	else:
		raise ValueError('Unsupported adversarial loss mode!')

	return loss.mean() if size_average else loss.sum()

def masked_prediction_loss(logits, target, mask, size_average, smooth=0):
	target = target[mask]
	mask_num = mask.float().sum(0, keepdim=True).expand_as(mask)[mask]
	lprobs = F.log_softmax(logits, -1)
	# masked_target = target[mask]
	loss = - lprobs.gather(1, target.unsqueeze(-1)).squeeze(-1)
	if smooth > 0:
		smooth_loss = -lprobs.sum(-1)
		eps = smooth / lprobs.size(-1)
		loss = (1-smooth)*loss + eps*smooth_loss

	# num_mask = mask.float().sum(0, keepdim=True).expand_as(mask)[mask]
	loss = loss / mask_num 
	loss = loss.sum()
	if size_average:
		loss = loss / mask.size(1)

	return loss

def masked_prediction_loss_reform(logits, target, mask, old_mask_reform, size_average, smooth=0):
	target = target[mask]
	mask_num = old_mask_reform.float().sum(0, keepdim=True).expand_as(mask)[mask]
	lprobs = F.log_softmax(logits, -1)
	# masked_target = target[mask]
	loss = - lprobs.gather(1, target.unsqueeze(-1)).squeeze(-1)
	if smooth > 0:
		smooth_loss = -lprobs.sum(-1)
		eps = smooth / lprobs.size(-1)
		loss = (1-smooth)*loss + eps*smooth_loss

	# num_mask = mask.float().sum(0, keepdim=True).expand_as(mask)[mask]
	loss = loss / mask_num
	loss = loss[old_mask_reform[mask]]
	loss = loss.sum()
	if size_average:
		loss = loss / mask.size(1)

	return loss

def masked_prediction_acc(logits, target, mask, size_average):
	with torch.no_grad():
		target = target[mask]
		mask_num = mask.float().sum(0, keepdim=True).expand_as(mask)[mask]
		pred = logits.argmax(-1)
		loss = (pred == target).float()
		loss = loss / mask_num
		loss = loss.sum()
		if size_average:
			loss = loss / mask.size(1)

	return loss.item()

def masked_softmax_mse_loss(src_logits, tgt_logits, tgt_mask, sym, size_average, tau, topk):
	# if not sym:
	# 	tgt_logits = tgt_logits.detach()
	
	mask_num = tgt_mask.float().sum(0, keepdim=True).expand_as(tgt_mask)[tgt_mask]
	src_logits = src_logits.view(-1, src_logits.size(-1))[tgt_mask.view(-1)]
	if topk is not None:
		tgt_logits, top_inds = tgt_logits.topk(topk, -1)
		src_logits = src_logits.gather(-1, top_inds)
	src_probs = F.softmax(src_logits, -1)
	tgt_probs = F.softmax(tgt_logits/tau, -1)
	loss = torch.mean((src_probs - tgt_probs) ** 2, -1)
	loss = loss / mask_num 
	loss = loss.sum()
	if size_average:
		loss = loss / tgt_mask.size(1)

	return loss

def masked_kl_loss(src_logits, tgt_logits, tgt_mask, sym, size_average, tau, topk):
	mask_num = tgt_mask.float().sum(0, keepdim=True).expand_as(tgt_mask)[tgt_mask]
	src_logits = src_logits.view(-1, src_logits.size(-1))[tgt_mask.view(-1)]
	if topk is not None:
		tgt_logits, top_inds = tgt_logits.topk(topk, -1)
		src_logits = src_logits.gather(-1, top_inds)
	src_lprobs = F.log_softmax(src_logits, -1)
	tgt_probs = F.softmax(tgt_logits.detach()/tau, -1)
	loss = F.kl_div(src_lprobs, tgt_probs, reduction='none')
	if sym:
		src_probs = F.softmax(src_logits.detach()/tau, -1)
		tgt_lprobs = F.log_softmax(tgt_logits, -1)
		loss = loss + F.kl_div(tgt_lprobs, src_probs, reduction='none')
	loss = loss.sum(-1)
	loss = loss / mask_num 
	loss = loss.sum()
	if size_average:
		loss = loss / tgt_mask.size(1)

	return loss

def consistency_loss(src_logits, tgt_logits, tgt_mask, sym, l2_mode, size_average, tau, topk):
	if l2_mode:
		return masked_softmax_mse_loss(src_logits, tgt_logits, tgt_mask, sym, size_average, tau, topk)
	else:
		return masked_kl_loss(src_logits, tgt_logits, tgt_mask, sym, size_average, tau, topk)


def consistency_full_loss(src_logits, tgt_logits, tgt_mask, sym, l2_mode, size_average, tau, topk, filling_mask=None):
	if filling_mask is not None:
		src_logits = shrink_3d(src_logits, filling_mask)
	if topk is not None:
		tgt_logits, top_inds = tgt_logits.topk(topk, -1)
		src_logits = src_logits.gather(-1, top_inds)
	if l2_mode:
		src_probs = F.softmax(src_logits, -1)
		tgt_probs = F.softmax(tgt_logits/tau, -1)
		loss = torch.mean((src_probs - tgt_probs) ** 2, -1)
	else:
		src_lprobs = F.log_softmax(src_logits, -1)
		tgt_probs = F.softmax(tgt_logits.detach()/tau, -1)
		loss = F.kl_div(src_lprobs, tgt_probs, reduction='none')
		if sym:
			src_probs = F.softmax(src_logits.detach()/tau, -1)
			tgt_lprobs = F.log_softmax(tgt_logits, -1)
			loss = loss + F.kl_div(tgt_lprobs, src_probs, reduction='none')
		loss = loss.sum(-1)

	loss = loss.masked_fill(~tgt_mask, 0)
	loss = loss.sum(0) / tgt_mask.float().sum(0)

	return loss.mean() if size_average else loss.sum()



def consistency_acc(src_logits, tgt_logits, tgt_mask, size_average):
	with torch.no_grad():
		mask_num = tgt_mask.float().sum(0, keepdim=True).expand_as(tgt_mask)[tgt_mask]
		tgt = tgt_logits.argmax(-1)
		src = src_logits.view(-1, src_logits.size(-1)).argmax(-1)[tgt_mask.view(-1)]
		loss = (src == tgt).float() / mask_num
		loss = loss.sum()
		if size_average:
			loss = loss / tgt_mask.size(1)

	return loss.item()

def coverage_score(attn, klens, qlens, k_mask_b, q_mask_b, mode, inverse, with_start):
	if with_start:
		qlens = qlens + 1
		q_mask_b = cat_zeros_at_start(q_mask_b)
	else:
		attn = attn[:, 1:, :]
	# non_zero_mask = (~k_mask_b.unsqueeze(1)) * (~q_mask_b.unsqueeze(-1))
	attn = attn.masked_fill(q_mask_b.unsqueeze(-1), 0)
	attn = attn.sum(1) / qlens.float().unsqueeze(-1) # bsz * klen

	if mode != 'ent':
		tgt_attn = (1 / klens.float()).unsqueeze(-1)
		if mode == 'l2':
			score = (attn - tgt_attn) ** 2
			score = score.masked_fill(k_mask_b, 0)
			score = score.sum(1) / klens.float()
		elif mode == 'l1':
			score = torch.abs(attn - tgt_attn)
			score = score.masked_fill(k_mask_b, 0)
			score = score.sum(1) / klens.float()
		elif mode == 'max':
			score = torch.abs(attn - tgt_attn).masked_fill(k_mask_b, 0)
			score = score.max(1)[0]
		elif mode == 'kl':
			attn_lp = torch.log(attn + 1e-10)
			score = F.kl_div(attn_lp, tgt_attn, reduction='none')
			score = score.masked_fill(k_mask_b, 0)
			score = score.sum(1)
		else:
			raise ValueError('Unsupported coverage mode!')
		score = - score if inverse else 1/score
	else:
		attn_lp = torch.log(attn + 1e-10)
		score = - attn * attn_lp
		score = score.masked_fill(k_mask_b, 0)
		score = score.sum(1)

	return score



def attn_loss(src_attn, tgt_attn, attn_name, klens, qlens, k_mask_b, q_mask_b, n_layers, sym, l2_mode, size_average):
	n = len(src_attn)
	assert n >= n_layers
	non_zero_mask = (~k_mask_b.unsqueeze(1)) * (~q_mask_b.unsqueeze(-1))
	loss = 0
	for i in range(n_layers):
		loss = loss + (
			mse_attn_loss(src_attn[n-1-i][attn_name], tgt_attn[n-1-i][attn_name], klens, qlens, non_zero_mask, sym, size_average) 
			if l2_mode else
			kl_attn_loss(src_attn[n-1-i][attn_name], tgt_attn[n-1-i][attn_name], qlens, non_zero_mask, sym, size_average)
			)
	return loss

def mse_attn_loss(src_attn, tgt_attn, klens, qlens, non_zero_mask, sym, size_average):
	# if not sym:
	# 	tgt_attn = tgt_attn.detach()
	loss = (src_attn - tgt_attn) ** 2
	loss = loss * non_zero_mask
	loss = loss.sum(-1) / klens.float().unsqueeze(-1)
	loss = loss.sum(-1) / qlens.float()
	return loss.mean() if size_average else loss.sum()

def kl_attn_loss(src_attn, tgt_attn, qlens, non_zero_mask, sym, size_average):
	src_attn_lp = torch.log(src_attn + 1e-10)
	loss = F.kl_div(src_attn_lp, tgt_attn.detach(), reduction='none')
	if sym:
		tgt_attn_lp = torch.log(tgt_attn + 1e-10)
		loss = loss + F.kl_div(tgt_attn_lp, src_attn.detach(), reduction='none')
	loss = loss * non_zero_mask
	loss = loss.sum((1, 2)) / qlens.float()
	return loss.mean() if size_average else loss.sum()

def get_discard_mask(x1, x2, lens1, lens2, padding_mask1, max_offset):
	max_len, bsz = x1.size()
	inds = torch.arange(0, max_len, dtype=torch.float, device=x1.device)
	one_tok_mask = lens1 == 1
	if one_tok_mask.any():
		lens1 = lens1.masked_fill(one_tok_mask, 2)
	inds = torch.round(inds.unsqueeze(-1) / (lens1 - 1).unsqueeze(0).float() * (lens2 - 1).unsqueeze(0).float()).long()
	n_offsets = max_offset * 2 + 1
	offsets = torch.arange(-max_offset, max_offset+1, dtype=torch.long, device=x1.device).unsqueeze(-1).unsqueeze(-1)
	inds = inds.unsqueeze(0) + offsets
	inds = torch.min(inds, (lens2-1).unsqueeze(0).unsqueeze(0)).clamp(min=0)#.masked_fill(padding_mask1.unsqueeze(0), 0)
	compx = x2.gather(0, inds.view(-1, bsz)).view(n_offsets, max_len, bsz)
	masks = x1.unsqueeze(0) == compx
	discard_mask = masks.long().sum(0) == 0
	discard_mask = discard_mask.masked_fill(padding_mask1, 0)
	return discard_mask

def under_trans_loss(logits, effective_mask, src, discard_mask, size_average):
	seq_len = logits.size(0)
	nwords, bsz = src.size()
	probs = F.softmax(logits, -1)
	selected = probs.gather(-1, src.t().unsqueeze(0).expand(seq_len, bsz, nwords)).masked_fill(~effective_mask.unsqueeze(-1), 0).sum(0)
	effective_lens = effective_mask.float().sum(0)
	loss = selected.masked_fill(~discard_mask.t(), 0).sum(-1) / effective_lens.clamp(min=1)
	return loss.mean() if size_average else loss.sum()



def compute_bleu_score(labels_files, predictions_path):

	if not isinstance(labels_files, list):
		labels_files = [labels_files]

	try:
		cmd = 'perl %s %s < %s' % ('./multi-bleu.perl',
								   " ".join(labels_files),
								   predictions_path)
		bleu_out = subprocess.check_output(
			cmd,
			stderr=subprocess.STDOUT,
			shell=True)
		bleu_out = bleu_out.decode("utf-8")
		bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
		return float(bleu_score)
	except subprocess.CalledProcessError as error:
		if error.output is not None:
			msg = error.output.strip()
			print("bleu script returned non-zero exit code: {}".format(msg))
		return 0
