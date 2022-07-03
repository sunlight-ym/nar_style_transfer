from collections import Counter
import math
from classifiers import *
import torch.nn.functional as F
import numpy as np

def at_as_eval(at_model, x, style, to_style, seq_mask_b, outputs, outputs_padding_mask_b):
	bsz = style.size(0)
	bsz_ex = outputs.size(1)
	beam_size = bsz_ex // bsz

	style_emb = at_model.style_emb_layer(style)
	to_style = repeat_tensors(beam_size, 0, to_style)
	to_style_emb = at_model.style_emb_layer(to_style)
	
	longest_enc_pos_emb = at_model.position_embed(x.size(0)+1, style.device, True).expand(x.size(0)+1, bsz, at_model.emb_size).contiguous()
	longest_dec_pos_emb = at_model.position_embed(outputs.size(0), style.device, False).expand(outputs.size(0), bsz_ex, at_model.emb_size).contiguous()
	style_enc_pos_emb, style_dec_pos_emb, longest_enc_pos_emb, longest_dec_pos_emb = longest_enc_pos_emb[0], longest_dec_pos_emb[0], longest_enc_pos_emb[1:], longest_dec_pos_emb[1:]
	style_enc_emb = style_emb + style_enc_pos_emb
	to_style_dec_emb = to_style_emb + style_dec_pos_emb

	enc_seq_emb = at_model.embed(x, longest_enc_pos_emb)
	enc_outputs = at_model.encode(enc_seq_emb, style_enc_emb, seq_mask_b)
	enc_outputs = repeat_tensors(beam_size, 1, enc_outputs)
	seq_mask_b = repeat_tensors(beam_size, 0, seq_mask_b)
	dec_seq_emb = at_model.embed(outputs[:-1], longest_dec_pos_emb)
	logits = at_model.para_decode(dec_seq_emb, to_style_dec_emb, enc_outputs, seq_mask_b, outputs_padding_mask_b, to_style)['logits']
	lprobs = F.log_softmax(logits, -1)
	lprobs = lprobs.gather(-1, outputs.unsqueeze(-1)).squeeze(-1)
	return lprobs

def reward(style_tool, to_style, inputs, inputs_lens, outputs, outputs_lens, outputs_padding_mask, outputs_padding_mask_b, beta):
	bsz = inputs.size(1)
	beam_size = outputs.size(1) // bsz
	to_style = repeat_tensors(beam_size, 0, to_style)
	batch_dict_for_clf = {}
	if isinstance(style_tool, cnn_classifier):
		batch_dict_for_clf['x_b'] = outputs.transpose(0, 1).contiguous()
		batch_dict_for_clf['padding_mask_b'] = outputs_padding_mask_b
		batch_dict_for_clf['x_b'] = batch_dict_for_clf['x_b'].masked_fill(batch_dict_for_clf['padding_mask_b'], constants.PAD_ID)
	elif isinstance(style_tool, attn_classifier):
		batch_dict_for_clf['x'] = outputs
		batch_dict_for_clf['padding_mask'] = outputs_padding_mask
	else:
		batch_dict_for_clf['x'] = outputs
		batch_dict_for_clf['padding_mask_b'] = outputs_padding_mask_b
		batch_dict_for_clf['lens'] = outputs_lens
	clf_result = style_tool(batch_dict_for_clf)
	clf_scores = F.softmax(clf_result, -1).gather(-1, to_style.unsqueeze(-1)).view(-1)

	inputs_arr = inputs.t().cpu().numpy()
	outputs_arr = outputs.t().cpu().view(bsz, beam_size, outputs.size(0)).numpy()
	inputs_lens_arr = (inputs_lens-1).cpu().numpy()
	outputs_lens_arr = (outputs_lens-1).view(bsz, beam_size).cpu().numpy()
	bleu_scores = []
	for in_sen, out_sens, li, los in zip(inputs_arr, outputs_arr, inputs_lens_arr, outputs_lens_arr):
		for out_sen, lo in zip(out_sens, los):
			bleu_scores.append(get_bleu(in_sen[:li], out_sen[:lo]))
	bleu_scores = torch.tensor(bleu_scores, dtype=torch.float, device=clf_scores.device)
	beta_square = beta ** 2

	scores = (1 + beta_square) * clf_scores * bleu_scores / (clf_scores + beta_square * bleu_scores)
	return scores







def bleu_stats(hypothesis, reference):
	"""Compute statistics for BLEU."""
	stats = []
	stats.append(len(hypothesis))
	stats.append(len(reference))
	for n in range(1, 5):
		s_ngrams = Counter(
			[tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
		)
		r_ngrams = Counter(
			[tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
		)
		stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
		stats.append(max([len(hypothesis) + 1 - n, 0]))
	return stats

def bleu(stats):
	"""Compute BLEU given n-gram statistics."""
	if len(list(filter(lambda x: x == 0, stats))) > 0:
		return 0
	(c, r) = stats[:2]
	log_bleu_prec = sum(
		[math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
	) / 4.
	return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)

# def get_bleu(hypotheses, references):
# 	"""Get validation BLEU score for dev set."""
# 	stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
# 	for hyp, ref in zip(hypotheses, references):
# 		stats += np.array(bleu_stats(hyp, ref))
# 	return 100 * bleu(stats)

def get_bleu(hyp, ref):
	return bleu(np.array(bleu_stats(hyp, ref)))