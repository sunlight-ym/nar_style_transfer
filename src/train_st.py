import time
import gc
import argparse
import sys
import os
import copy
from collections import namedtuple, OrderedDict
from itertools import chain
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from datalib.batch import Batch
from datalib.iterator import BucketIterator
from data_loader import *
from train_utils import *
from loss import *
from layers import get_padding_mask, reverse_seq
from models import style_transfer, style_transfer_transformer
from classifiers import *
from lms import *
from search import SequenceGenerator
from torch.utils.tensorboard import SummaryWriter
import datalib.constants as constants
#torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark=True



class Solver(object):
	"""docstring for Solver"""
	def __init__(self, config):
		super(Solver, self).__init__()
		self.debug = config.debug
		self.eval_only = config.eval_only
		self.keep_only_full = config.keep_only_full
		self.start_saving = config.start_saving
		
		self.use_transformer = config.use_transformer
		self.batch_merge = config.batch_merge
		self.opt_accross_styles = config.opt_accross_styles
		self.tau = config.tau
		self.tau_up = config.tau_up
		self.tau_update_start = config.tau_update_start
		self.tau_update_end = config.tau_update_end
		self.greedy_train = config.greedy_train
		self.trans_extra_len = config.trans_extra_len
		self.trans_extra_len_eval = config.trans_extra_len_eval
		self.eval_beam_search = config.eval_beam_search
		self.eval_dropout = config.eval_dropout
		self.beam_size = config.beam_size
		
		self.rec_weight = config.rec_weight
		self.pseudo_weight = config.pseudo_weight
		self.simul_weight = config.simul_weight
		self.bt_weight = config.bt_weight
		self.clf_weight = config.clf_weight
		self.lm_weight = config.lm_weight
		self.enc_pred_weight = config.enc_pred_weight
		self.clf_adv = config.clf_adv
		self.clf_adv_mode = config.clf_adv_mode
		self.clf_adv_scale = config.clf_adv_scale
		self.clf_adv_src_scale = config.clf_adv_src_scale
		
		self.smooth = config.smooth
		self.lm_tgt_mode = config.lm_tgt_mode
		self.bt_sg = config.bt_sg
		self.less_len = config.less_len
		if self.enc_pred_weight > 0:
			assert config.mask_src_rate > 0
		
		self.mono = self.bt_weight > 0 or self.clf_weight > 0 or self.lm_weight > 0 or self.enc_pred_weight > 0
		
		self.weight_up_start = config.weight_up_start
		self.weight_up_end = config.weight_up_end
		self.weight_down_start = config.weight_down_start
		self.weight_down_end = config.weight_down_end
		self.rec_down = config.rec_down
		self.pseudo_down = config.pseudo_down
		self.simul_down = config.simul_down
		
		self.max_grad_norm = config.max_grad_norm
		self.update_interval = config.update_interval
		self.up_alpha = config.up_alpha
		self.down_alpha = config.down_alpha
		self.n_iters = config.n_iters
		self.log_interval = config.log_interval
		self.eval_interval = config.eval_interval		
		self.device = torch.device('cuda', config.gpu)
		self.save_start = config.save_start
		self.save_interval = config.save_interval
		
		self.para_top_num = config.para_top_num

		if config.save_para_train:

			pseudo_path_dir = os.path.join(config.work_dir, 'pseudo', config.dataset)
			makedirs(pseudo_path_dir)
			self.pseudo_path = os.path.join(pseudo_path_dir, 'trans-{}{}'.format(config.version, 'b' if self.eval_beam_search else ''))
		else:

			self.model_path = os.path.join(config.work_dir, 'model', config.dataset, 'trans', config.version)
			makedirs(self.model_path)
			self.output_path = os.path.join(config.work_dir, 'output', config.dataset, 'trans', config.version)
			makedirs(self.output_path)
			if not self.eval_only:
				self.summary_path = os.path.join(config.work_dir, 'summary', config.dataset, 'trans', config.version)
				makedirs(self.summary_path)
				self.summary_writer = SummaryWriter(self.summary_path)
		

		
		if self.clf_weight > 0:
			check_point = torch.load(config.style_train_tool, map_location=lambda storage, loc: storage)
			tool_type = check_point['model_type']
			tool_type = {'cnn': cnn_classifier, 'rnn': attn_classifier, 'sa': Transformer_classifier}[tool_type]
			self.style_train_tool = tool_type(*check_point['args'])
			self.style_train_tool.to(self.device)
			self.style_train_tool.load_state_dict(check_point['model_state'])
			del check_point
			# self.style_train_tool = torch.load(config.style_train_tool, map_location=self.device)['model']
			if self.clf_adv and self.clf_adv_mode=='ac':
				add_one_class(self.style_train_tool)
			if self.clf_adv:
				self.style_train_tool_update = copy.deepcopy(self.style_train_tool)
				self.style_train_tool_update.train()
				self.clf_adv_optimizer = build_optimizer(config.optim_method, self.style_train_tool_update, 
					config.clf_adv_lr, config.momentum, config.weight_decay, config.beta2)
			frozen_model(self.style_train_tool)
			if config.aux_model_eval_mode:
				self.style_train_tool.eval()
			else:
				self.style_train_tool.train()

		if self.lm_weight > 0:
			check_point = torch.load(config.fluency_train_tool, map_location=lambda storage, loc: storage)
			tool_type = check_point['model_type']
			tool_type = {'rnn': rnn_lm, 'sa': Transformer_lm}[tool_type]
			self.fluency_train_tool = tool_type(*check_point['args'])
			self.fluency_train_tool.to(self.device)
			self.fluency_train_tool.load_state_dict(check_point['model_state'])
			del check_point
			frozen_model(self.fluency_train_tool)
			if config.aux_model_eval_mode:
				self.fluency_train_tool.eval()
			else:
				self.fluency_train_tool.train()

		# load evaluators
		check_point = torch.load(config.style_eval_tool, map_location=lambda storage, loc: storage)
		tool_type = check_point['model_type']
		tool_type = {'cnn': cnn_classifier, 'rnn': attn_classifier, 'sa': Transformer_classifier}[tool_type]
		self.style_eval_tool = tool_type(*check_point['args'])
		self.style_eval_tool.to(self.device)
		self.style_eval_tool.load_state_dict(check_point['model_state'])
		del check_point
		# self.style_eval_tool = torch.load(config.style_eval_tool, map_location=self.device)['model']
		frozen_model(self.style_eval_tool)
		self.style_eval_tool.eval()

		check_point = torch.load(config.fluency_eval_tool, map_location=lambda storage, loc: storage)
		tool_type = check_point['model_type']
		tool_type = {'rnn': rnn_lm, 'sa': Transformer_lm}[tool_type]
		self.fluency_eval_tool = tool_type(*check_point['args'])
		self.fluency_eval_tool.to(self.device)
		self.fluency_eval_tool.load_state_dict(check_point['model_state'])
		del check_point
		# self.fluency_eval_tool = torch.load(config.fluency_eval_tool, map_location=self.device)['model']
		frozen_model(self.fluency_eval_tool)
		self.fluency_eval_tool.eval()

		self.bi_fluency_eval = config.fluency_eval_tool_rev is not None
		if self.bi_fluency_eval:
			check_point = torch.load(config.fluency_eval_tool_rev, map_location=lambda storage, loc: storage)
			tool_type = check_point['model_type']
			tool_type = {'rnn': rnn_lm, 'sa': Transformer_lm}[tool_type]
			self.fluency_eval_tool_rev = tool_type(*check_point['args'])
			self.fluency_eval_tool_rev.to(self.device)
			self.fluency_eval_tool_rev.load_state_dict(check_point['model_state'])
			del check_point
			# self.fluency_eval_tool_rev = torch.load(config.fluency_eval_tool_rev, map_location=self.device)['model']
			frozen_model(self.fluency_eval_tool_rev)
			self.fluency_eval_tool_rev.eval()


		datasets = Corpus.iters_dataset_simple(config.work_dir, config.dataset, 'trans_new', config.max_sen_len, 
							config.min_freq, config.max_vocab_size, noisy=self.rec_weight>0, 
							noise_drop=config.noise_drop, noise_drop_as_unk=config.noise_drop_as_unk, 
							noise_insert=config.noise_insert, noise_insert_self=config.noise_insert_self,
							mask_src_rate=config.mask_src_rate, mask_src_consc=config.mask_src_consc, 
							mask_src_span=config.mask_src_span, mask_src_span_len=config.mask_src_span_len,
							with_pseudo=self.pseudo_weight>0 or self.simul_weight>0, pseudo_path=config.pseudo_path)
		if config.sub < 1:
			old_len = len(datasets[0])
			random.shuffle(datasets[0].examples)
			new_len = int(old_len * config.sub)
			datasets[0].examples = datasets[0].examples[:new_len]
			print('cutting the training data from {} to {}'.format(old_len, len(datasets[0])))
		self.valid_loader, self.test_loader = BucketIterator.splits(datasets[1:], 
							batch_size = config.eval_batch_size, 
							train_flags = [False]*2)
		if config.save_para_train:
			self.train_loaders = BucketIterator.splits(datasets[:1], 
							batch_size = config.eval_batch_size, 
							train_flags = [False])
		else:

			trainsets = datasets[0].stratify_split('label') if config.label_split else (datasets[0],)
			self.train_loaders = BucketIterator.splits(trainsets, 
							batch_size = config.batch_size, 
							train_flags = [True]*len(trainsets))
		

		vocab_size = len(self.train_loaders[0].dataset.fields['text'].vocab)
		num_classes = len(self.train_loaders[0].dataset.fields['label'].vocab)
		self.num_classes = num_classes
		self.dataset_stats = {'test':get_class_stats(datasets[2], 'label'), 'valid':get_class_stats(datasets[1], 'label')}
		self.human_files = get_human_files(config.work_dir, config.dataset, num_classes, config.num_ref)
		self.test_files = get_test_files(config.work_dir, config.dataset, num_classes)

		print('number of human files:', len(self.human_files))
		print('number of test files:', len(self.test_files))

		for i in range(len(self.train_loaders)):
			print('train size', i, ':', len(self.train_loaders[i].dataset))
		print('valid size:', len(self.valid_loader.dataset))
		print('test size:', len(self.test_loader.dataset))
		print('vocab size:', vocab_size)
		print('number of classes:', num_classes)

		self.itos = self.train_loaders[0].dataset.fields['text'].vocab.itos
		# self.label_itos = self.train_loaders[0].fields['label'].vocab.itos
		
		if not self.use_transformer:
			self.args = [vocab_size, config.emb_size, config.emb_max_norm,
				config.rnn_type, config.hid_size, True, config.dec_hid_size, config.enc_num_layers, config.dec_num_layers, config.pooling_size,
				config.h_only, config.diff_bias, num_classes, config.feed_last_context, config.use_att, config.enc_cat_style]
			self.model = style_transfer(vocab_size, config.emb_size, config.emb_max_norm,
				config.rnn_type, config.hid_size, True, config.dec_hid_size, config.enc_num_layers, config.dec_num_layers, config.pooling_size,
				config.h_only, config.diff_bias, num_classes, config.feed_last_context, config.use_att, config.enc_cat_style)
		else:
			self.args = [vocab_size, config.emb_size, config.emb_max_norm, config.pos_sincode, config.token_emb_scale, config.max_sen_len,
				config.dropout_rate, config.num_heads, config.hid_size, config.enc_num_layers, config.dec_num_layers,
				config.diff_bias, num_classes,
				config.att_dropout_rate, config.transformer_norm_bf, config.enc_cat_style, config.share_pos_emb]
			self.model = style_transfer_transformer(vocab_size, config.emb_size, config.emb_max_norm, config.pos_sincode, config.token_emb_scale, config.max_sen_len,
				config.dropout_rate, config.num_heads, config.hid_size, config.enc_num_layers, config.dec_num_layers,
				config.diff_bias, num_classes,
				config.att_dropout_rate, config.transformer_norm_bf, config.enc_cat_style, config.share_pos_emb)
		
		if config.pretrained_emb is not None:
			text_vocab = self.train_loaders[0].dataset.fields['text'].vocab
			text_vocab.load_vectors(config.pretrained_emb, cache=os.path.join(config.work_dir, 'word_vectors'), max_vectors=config.pretrained_emb_max)
			self.model.emb_layer.weight.data.copy_(text_vocab.vectors)
			text_vocab.vectors = None
		self.model.to(self.device)
		self.optimizer = build_optimizer(config.optim_method, self.model, config.lr, config.momentum, config.weight_decay, config.beta2)
		self.lr_scheduler = build_lr_scheduler(config.lr_method, self.optimizer, config.lr_warmup_steps, 
												config.lr_decay_steps, config.lr_decay_mode, config.lr_min_factor, config.lr_decay_rate, config.lr_init_factor)
		self.step = 1
		if config.train_from is not None:
			check_point=torch.load(config.train_from, map_location=lambda storage, loc: storage)
			self.model.load_state_dict(check_point['model_state'])
			if self.clf_weight > 0 and self.clf_adv:
				self.style_train_tool_update.load_state_dict(check_point['clf_model_state'])
				self.style_train_tool.load_state_dict(check_point['clf_model_state'])
			if config.load_optim:
				self.optimizer.load_state_dict(check_point['optimizer_state'])
				self.lr_scheduler.load_state_dict(check_point['lr_scheduler_state'])
				if self.clf_weight > 0 and self.clf_adv:
					self.clf_adv_optimizer.load_state_dict(check_point['clf_adv_optimizer_state'])
				
			self.step = check_point['step'] + 1
			del check_point
		
		if self.eval_beam_search:
			self.beam_decoder = SequenceGenerator(self.beam_size, self.trans_extra_len_eval, config.min_len, config.less_len)


	def save_states(self, prefix = ''):
		check_point = {
			'args': self.args,
			'use_transformer': self.use_transformer,
			'step': self.step,
			'model_state': self.model.state_dict(),
			'optimizer_state': self.optimizer.state_dict(),
			'lr_scheduler_state': self.lr_scheduler.state_dict()
		}
		if self.clf_weight > 0 and self.clf_adv:
			check_point['clf_model_state'] = self.style_train_tool_update.state_dict()
			check_point['clf_adv_optimizer_state'] = self.clf_adv_optimizer.state_dict()
		
		filename = os.path.join(self.model_path, '{}model-{}'.format(prefix, self.step))
		torch.save(check_point, filename)

	# def save_model(self):
	# 	check_point = {
	# 		'model': self.model
	# 	}
	# 	filename = os.path.join(self.model_path, 'full-model-{}'.format(self.step))
	# 	torch.save(check_point, filename)
		
	def prepare_batch(self, batch):
		xc, cenc_lens, x, enc_lens, mx_enc_in, mx_enc_out = [
			batch.text.get(k) for k in ['xc', 'cenc_lens', 'x', 'enc_lens', 'mx_enc_in', 'mx_enc_out']]
		style = batch.label
		mx_enc_mask = (mx_enc_out != constants.PAD_ID) if mx_enc_in is not None and self.enc_pred_weight > 0 else None
		
		seq_mask = get_padding_mask(x, enc_lens)
		seq_mask_b = seq_mask.t().contiguous() if self.use_transformer or (
			self.clf_weight > 0 and self.clf_adv and not isinstance(self.style_train_tool, attn_classifier)) else None

		cseq_mask = get_padding_mask(xc, cenc_lens) if xc is not None else None
		cseq_mask_b = cseq_mask.t().contiguous() if xc is not None and self.use_transformer else None
		if hasattr(batch, 'pseudo'):
			px, penc_lens = batch.pseudo['x'], batch.pseudo['enc_lens']
			pseq_mask = get_padding_mask(px, penc_lens)
			pseq_mask_b = pseq_mask.t().contiguous() if self.use_transformer else None
		else:
			px, penc_lens, pseq_mask, pseq_mask_b = None, None, None, None
		
		return {'xc': xc, 'x': x, 'enc_lens': enc_lens, 
				'seq_mask': seq_mask, 'seq_mask_b': seq_mask_b, 'style': style,
				'px': px, 'penc_lens': penc_lens, 'pseq_mask': pseq_mask, 'pseq_mask_b': pseq_mask_b,
				'cenc_lens': cenc_lens, 'cseq_mask': cseq_mask, 'cseq_mask_b': cseq_mask_b,
				'mx_enc_in': mx_enc_in, 'mx_enc_mask': mx_enc_mask}
	
	def prepare_input_for_clf(self, clf_model, result, soft):
		batch_dict_for_clf = {}
		if isinstance(clf_model, cnn_classifier):
			batch_dict_for_clf['x_b'] = result['soft_outputs' if soft else 'hard_outputs'].transpose(0, 1).contiguous()
			batch_dict_for_clf['padding_mask_b'] = result['hard_outputs_padding_mask_t_with_eos'] if self.use_transformer else result['hard_outputs_padding_mask_with_eos'].t().contiguous()
		elif isinstance(clf_model, attn_classifier):
			batch_dict_for_clf['x'] = result['soft_outputs' if soft else 'hard_outputs']
			batch_dict_for_clf['padding_mask'] = result['hard_outputs_padding_mask_with_eos']
		else:
			batch_dict_for_clf['x'] = result['soft_outputs' if soft else 'hard_outputs']
			batch_dict_for_clf['padding_mask_b'] = result['hard_outputs_padding_mask_t_with_eos'] if self.use_transformer else result['hard_outputs_padding_mask_with_eos'].t().contiguous()
			batch_dict_for_clf['lens'] = result['hard_outputs_lens_with_eos']
		return batch_dict_for_clf

	def prepare_input_for_clf_adv(self, clf_model, batch_dict):
		batch_dict_for_clf = {}
		input_text = batch_dict['x']
		if isinstance(clf_model, cnn_classifier):
			batch_dict_for_clf['x_b'] = input_text.t().contiguous()
			batch_dict_for_clf['padding_mask_b'] = batch_dict['seq_mask_b']
		elif isinstance(clf_model, attn_classifier):
			batch_dict_for_clf['x'] = input_text
			batch_dict_for_clf['padding_mask'] = batch_dict['seq_mask']
		else:
			batch_dict_for_clf['x'] = input_text
			batch_dict_for_clf['padding_mask_b'] = batch_dict['seq_mask_b']
			batch_dict_for_clf['lens'] = batch_dict['enc_lens']
		return batch_dict_for_clf

	def compute_loss(self, para_result, mono_result, pseudo_result, simul_result, batch_dict, size_average):
		loss_values = OrderedDict()
		x, lens, seq_mask, px, plens, pseq_mask, style, mx_enc_mask = [
			batch_dict[k] for k in ['x', 'enc_lens', 'seq_mask', 'px', 'penc_lens', 'pseq_mask', 'style', 'mx_enc_mask']]

		loss_all = 0
		down_weight = rampdown(self.step, self.weight_down_start, self.weight_down_end, self.update_interval, self.down_alpha, True)
		
		if para_result is not None:
			
			scale = (down_weight if self.rec_down else 1) * self.rec_weight
			rec_loss = seq_ce_logits_loss(para_result['logits'], x, lens, seq_mask, size_average, smooth=self.smooth)
			loss_values['rec'] = rec_loss.item()
			loss_all = loss_all + scale * rec_loss
			loss_values['rec_acc'] = seq_acc(para_result['logits'], x, lens, seq_mask, size_average)
		if pseudo_result is not None:
			scale = (down_weight if self.pseudo_down else 1) * self.pseudo_weight
			pseudo_loss = seq_ce_logits_loss(pseudo_result['logits'], x, lens, seq_mask, size_average, smooth=self.smooth)
			loss_values['pseudo'] = pseudo_loss.item()
			loss_all = loss_all + scale * pseudo_loss
			loss_values['pseudo_acc'] = seq_acc(pseudo_result['logits'], x, lens, seq_mask, size_average)
		if simul_result is not None:
			scale = (down_weight if self.simul_down else 1) * self.simul_weight
			simul_loss = seq_ce_logits_loss(simul_result['logits'], px, plens, pseq_mask, size_average, smooth=self.smooth)
			loss_values['simul'] = simul_loss.item()
			loss_all = loss_all + scale * simul_loss
			loss_values['simul_acc'] = seq_acc(simul_result['logits'], px, plens, pseq_mask, size_average)
		if mono_result is not None:
			up_weight = rampup(self.step, self.weight_up_start, self.weight_up_end, self.update_interval, self.up_alpha, True)

			if 'fw' in mono_result:
			
				loss_values['zl_rate'] = mono_result['fw']['hard_output_zl_mask'].float().mean().item() if size_average else mono_result['fw']['hard_output_zl_mask'].float().sum().item()
			
			if self.bt_weight > 0:
				scale = self.bt_weight * up_weight
				
				bt_loss = seq_ce_logits_loss(mono_result['bw']['logits'], x, lens, seq_mask, size_average, 
											batch_mask=mono_result['fw']['hard_output_zl_mask'], 
											smooth=self.smooth)
				loss_values['bt'] = bt_loss.item()
				loss_values['bt_acc'] = seq_acc(mono_result['bw']['logits'], x, lens, seq_mask, size_average)
				loss_all = loss_all + scale * bt_loss
			if self.enc_pred_weight > 0:
				scale = self.enc_pred_weight * up_weight
				enc_pred_loss = masked_prediction_loss(mono_result['enc_logits'], x, mx_enc_mask, size_average)
				loss_values['enc_pred'] = enc_pred_loss.item()
				loss_values['enc_pred_acc'] = masked_prediction_acc(mono_result['enc_logits'], x, mx_enc_mask, size_average)
				loss_all = loss_all + scale * enc_pred_loss
					
			if self.clf_weight > 0:
				scale = self.clf_weight * up_weight
				style_logits = self.style_train_tool(self.prepare_input_for_clf(self.style_train_tool, mono_result['fw'], True), soft_input = True)
				clf_loss = F.cross_entropy(style_logits, mono_result['to_style'], reduction = 'mean' if size_average else 'sum')
				loss_values['clf'] = clf_loss.item()
				loss_all = loss_all + scale * clf_loss
				loss_values['style_acc'] = unit_acc(style_logits, mono_result['to_style'], size_average)
				
				if self.clf_adv:
					src_logits = self.style_train_tool_update(self.prepare_input_for_clf_adv(self.style_train_tool, batch_dict))
					clf_adv_src_loss = F.cross_entropy(src_logits, style, reduction = 'mean' if size_average else 'sum')
					loss_values['clf_adv_src'] = clf_adv_src_loss.item()
					loss_all = loss_all + scale * self.clf_adv_scale * self.clf_adv_src_scale * clf_adv_src_loss
					loss_values['clf_adv_src_acc'] = unit_acc(src_logits, style, size_average)
					
					tsf_logits = self.style_train_tool_update(self.prepare_input_for_clf(self.style_train_tool, mono_result['fw'], False))
					clf_adv_tsf_loss = adv_loss(tsf_logits, style, mono_result['to_style'], self.clf_adv_mode, size_average)
					loss_values['clf_adv_tsf'] = clf_adv_tsf_loss.item()
					loss_all = loss_all + scale * self.clf_adv_scale * clf_adv_tsf_loss
					loss_values['clf_adv_tsf_acc'] = unit_acc(tsf_logits, mono_result['to_style'], size_average)
			if self.lm_weight > 0:
				scale = self.lm_weight * up_weight
				soft_outputs = mono_result['fw']['soft_outputs']
				lens_with_eos = mono_result['fw']['hard_outputs_lens_with_eos']
				padding_mask_with_eos = mono_result['fw']['hard_outputs_padding_mask_with_eos']
				start_tok = soft_outputs.new_zeros((1, soft_outputs.size(1), soft_outputs.size(2)))
				start_tok[:, :, constants.BOS_ID] = 1
				lm_logits = self.fluency_train_tool({'x': torch.cat([start_tok, soft_outputs[:-1]], 0), 'inds': mono_result['to_style']}, True)
				lm_tgt = mono_result['fw']['hard_outputs'] if self.lm_tgt_mode == 'hard' else (soft_outputs if self.lm_tgt_mode == 'soft' else soft_outputs.detach())
				lm_loss = seq_ce_logits_loss(lm_logits, lm_tgt, lens_with_eos, padding_mask_with_eos, size_average)
				loss_values['lm'] = lm_loss.item()
				loss_all = loss_all + scale * lm_loss
				loss_values['ppl'] = math.exp(loss_values['lm'])



					
		loss_values['loss_total'] = loss_all.item()
		return loss_all, loss_values

	def train_batch(self, batch, para, pseudo, simul, mono, enc_pred, bt):
		b0 = debug_time_msg(self.debug)
		
		batch_dict = self.prepare_batch(batch)
		# self.optimizer.zero_grad()
		b1 = debug_time_msg(self.debug, b0, 'prepare batch')
		
		tau = update_arg(self.tau, self.tau_up, self.step, self.update_interval, self.tau_update_start, self.tau_update_end, self.up_alpha, self.down_alpha)
		para_result, mono_result, pseudo_result, simul_result = self.model.transfer(batch_dict, para, pseudo, simul, mono, enc_pred, bt, self.bt_sg, tau, self.greedy_train, self.trans_extra_len, self.less_len)
		b2 = debug_time_msg(self.debug, b1, 'forward')
		
		loss, loss_values = self.compute_loss(para_result, mono_result, pseudo_result, simul_result, batch_dict, True)
		b3 = debug_time_msg(self.debug, b2, 'computing loss')

		loss.backward()
		b4 = debug_time_msg(self.debug, b3, 'backward')
		# clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
		# self.optimizer.step()
		# add_to_writer(loss_values, self.step, 'train', self.summary_writer)

		return loss_values



	def train(self):
		start = time.time()
		def prepare_optimize(clf_adv_flag, loss_accl):
			self.optimizer.zero_grad()
			if clf_adv_flag:
				self.clf_adv_optimizer.zero_grad()
			loss_accl.clear()
		def optimize(clf_adv_flag, loss_accl, num_batches):
			if self.max_grad_norm is not None:
				clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
				if clf_adv_flag:
					clip_grad_norm_(self.style_train_tool_update.parameters(), self.max_grad_norm)
			
			self.optimizer.step()
			if clf_adv_flag:
				self.clf_adv_optimizer.step()
				self.style_train_tool.load_state_dict(self.style_train_tool_update.state_dict())
			self.lr_scheduler.step()

			if num_batches > 1:
				get_final_accl(loss_accl, num_batches)
			add_to_writer(loss_accl, self.step, 'train', self.summary_writer)
			if self.step % self.log_interval == 0:
				print('step [{}/{}] {:.2f} s elapsed'. format(self.step, self.n_iters, time.time() - start))
				print_loss(loss_accl)
			if self.step % self.eval_interval == 0:
				self.eval('valid', self.valid_loader)
				
				if (self.start_saving is None or self.step > self.start_saving):
					self.eval('test', self.test_loader, True)
					
				self.save_states('latest-')
				

		self.model.train()
		data_iters = [iter(tl) for tl in self.train_loaders]
		accl = OrderedDict()
		never_set_flag = True
		while self.step <= self.n_iters:
			if never_set_flag and self.rec_down and self.rec_weight > 0 and self.step > self.weight_down_end:
				never_set_flag = False
				self.train_loaders[0].dataset.fields['text'].postprocessing.turn_off_noise()

			
			para = self.rec_weight > 0 and not (self.rec_down and self.step > self.weight_down_end)
			pseudo = self.pseudo_weight > 0 and not (self.pseudo_down and self.step > self.weight_down_end)
			simul = self.simul_weight > 0 and not (self.simul_down and self.step > self.weight_down_end)
			up_stage = (self.weight_up_start is None or self.step > self.weight_up_start)
			mono = up_stage and self.mono
			bt = up_stage and self.bt_weight > 0
			enc_pred = up_stage and self.enc_pred_weight > 0
			clf_adv = up_stage and self.clf_weight > 0 and self.clf_adv
			ignore_fields = ['pseudo'] if not (pseudo or simul) else []

			
			if self.batch_merge:
				prepare_optimize(clf_adv, accl)
				update_flag = True
				b0 = debug_time_msg(self.debug)
				batch = [next(data_iter) for data_iter in data_iters]
				batch = list(chain.from_iterable(batch))
				batch = Batch(batch, self.train_loaders[0].dataset, self.device, ignore_fields)
				b1 = debug_time_msg(self.debug, b0, 'read data')
				try:
					loss_values = self.train_batch(batch, para, pseudo, simul, mono, enc_pred, bt)
					update_accl(accl, loss_values)
				except RuntimeError as e:
					if ('out of memory' in str(e)):
						print('step {} | WARNING: {}; skipping batch; redoing this step'.format(self.step, str(e)))
						update_flag = False
						gc.collect()
						torch.cuda.empty_cache()
					else:
						raise e
				if update_flag:
					optimize(clf_adv, accl, 1)
					self.step += 1

			else:
				if self.opt_accross_styles:
					prepare_optimize(clf_adv, accl)
				update_flag = True
				i = 0
				while i < len(data_iters):
					if not self.opt_accross_styles:
						prepare_optimize(clf_adv, accl)
					b0 = debug_time_msg(self.debug)
					
					batch = next(data_iters[i])
					batch = Batch(batch, self.train_loaders[i].dataset, self.device, ignore_fields)
					b1 = debug_time_msg(self.debug, b0, 'read data')
					try:
						loss_values = self.train_batch(batch, para, pseudo, simul, mono, enc_pred, bt)
						update_accl(accl, loss_values)
					except RuntimeError as e:
						if ('out of memory' in str(e)):
							print('step {} | WARNING: {}; skipping batch; redoing this step'.format(self.step, str(e)))
							update_flag = False
							gc.collect()
							torch.cuda.empty_cache()
							if self.opt_accross_styles:
								break
							else:
								continue
						else:
							raise e
					if not self.opt_accross_styles:
						optimize(clf_adv, accl, 1)
						self.step += 1
					i += 1
				if self.opt_accross_styles and update_flag:
					optimize(clf_adv, accl, len(data_iters))
					self.step += 1
		self.summary_writer.close()

	def fluency_eval(self, result, to_style):
		hard_outputs = result['hard_outputs']
		hard_outputs_lens_with_eos = result['hard_outputs_lens_with_eos']
		hard_outputs_padding_mask_with_eos = get_padding_mask(hard_outputs, hard_outputs_lens_with_eos)
		bsz = hard_outputs.size(1)
		start_tok = hard_outputs.new_full((1, bsz), constants.BOS_ID)
		fluency_eval_nll = seq_ce_logits_loss(
			self.fluency_eval_tool({'x': torch.cat([start_tok, hard_outputs[:-1]], 0), 'inds': to_style}), 
			hard_outputs, hard_outputs_lens_with_eos, hard_outputs_padding_mask_with_eos, False).item()
		if self.bi_fluency_eval:
			hard_outputs_rev = reverse_seq(hard_outputs, result['hard_outputs_lens'], hard_outputs==constants.EOS_ID)
			fluency_eval_nll = fluency_eval_nll + seq_ce_logits_loss(
				self.fluency_eval_tool_rev({'x': torch.cat([start_tok, hard_outputs_rev[:-1]], 0), 'inds': to_style}), 
				hard_outputs_rev, hard_outputs_lens_with_eos, hard_outputs_padding_mask_with_eos, False).item()
			fluency_eval_nll = fluency_eval_nll / 2
		return fluency_eval_nll

	def eval(self, name, dataset_loader, with_truth = False):
		if not self.eval_dropout:
			self.model.eval()
		n_total = len(dataset_loader.dataset)
		accl = OrderedDict()
		start = time.time()
		trans_sens = []
		self.train_loaders[0].dataset.fields['text'].postprocessing.to_eval_mode()
		ignore_fields = ['pseudo']
		
		with torch.no_grad():
			for i, batch in enumerate(dataset_loader):
				batch_dict = self.prepare_batch(Batch(batch, dataset_loader.dataset, self.device, ignore_fields))
				tau = update_arg(self.tau, self.tau_up, self.step, self.update_interval, self.tau_update_start, self.tau_update_end, self.up_alpha, self.down_alpha)
				result = self.model.transfer(batch_dict, False, False, False, True, False, False, None, tau, True, self.trans_extra_len_eval, self.less_len,
					self.beam_decoder if self.eval_beam_search else None)[1]
				
				style_eval_acc = unit_acc(self.style_eval_tool(self.prepare_input_for_clf(self.style_eval_tool, result['fw'], False)), 
					result['to_style'], False)
				fluency_eval_nll = self.fluency_eval(result['fw'], result['to_style'])
				update_accl(accl, {'style_eval_acc': style_eval_acc, 'fluency_eval_nll': fluency_eval_nll})
				to_sentences(result['fw']['hard_outputs'], result['fw']['hard_outputs_lens'], self.itos, trans_sens)
					
					
		get_final_accl(accl, n_total)
		accl['fluency_eval_nppl'] = - math.exp(accl['fluency_eval_nll'])
		
		src_sens = list(dataset_loader.dataset.text)
		print('src example (index 0 out of {}): {}'.format(len(src_sens), src_sens[0]))

		trans_sens = reorder(dataset_loader.order, trans_sens)
		print('trans example (index 0 out of {}): {}'.format(len(trans_sens), trans_sens[0]))
		result_file = os.path.join(self.output_path, '{}-result-{}'.format(name, self.step))
		save_results(src_sens, trans_sens, result_file)
		
		
		accl['self_bleu'], accl['human_bleu'] = self.save_outputs_and_compute_bleu(name, trans_sens, with_truth)
		append_scores(result_file, accl['style_eval_acc'], -accl['fluency_eval_nppl'], accl['self_bleu'], accl['human_bleu'])
		

		if not self.eval_only:
			add_to_writer(accl, self.step, name, self.summary_writer)
		print('{} performance of model at step {}'.format(name, self.step))
		print_loss(accl)
		print('{:.2f} s for evaluation'.format(time.time() - start))
		if not self.eval_dropout:
			self.model.train()
		self.train_loaders[0].dataset.fields['text'].postprocessing.to_train_mode()
		

	def save_outputs_and_compute_bleu(self, name, trans_sens, with_truth):
		stats = self.dataset_stats[name]
		cur_ind = 0
		self_bleu = 0
		if with_truth:
			human_bleu = 0
		
		for i in range(self.num_classes):
			output_file = os.path.join(self.output_path, '{}-result-{}.{}'.format(name, self.step, i))
			save_outputs(trans_sens, cur_ind, cur_ind+stats[i], output_file)
			self_bleu += compute_bleu_score(self.test_files[i], output_file)
			if with_truth:
				human_bleu += compute_bleu_score(self.human_files[i], output_file)
			cur_ind = cur_ind + stats[i]
			if self.keep_only_full:
				os.remove(output_file)
		self_bleu /= self.num_classes
		if with_truth:
			human_bleu /= self.num_classes
		return (self_bleu, human_bleu) if with_truth else (self_bleu, None) 

	def generate_parallel(self):
		if not self.eval_dropout:
			self.model.eval()
		self.train_loaders[0].dataset.fields['text'].postprocessing.to_eval_mode()
		ignore_fields = ['pseudo']
		dataset_loader = self.train_loaders[0]
		start = time.time()
		trans_sens = []
		# prob_list, ind_list = [], []
		with torch.no_grad():
			for i, batch in enumerate(dataset_loader):
				batch_dict = self.prepare_batch(Batch(batch, dataset_loader.dataset, self.device, ignore_fields))
				tau = update_arg(self.tau, self.tau_up, self.step, self.update_interval, self.tau_update_start, self.tau_update_end, self.up_alpha, self.down_alpha)
				result = self.model.transfer(batch_dict, False, False, False, True, False, False, None, tau, True, self.trans_extra_len_eval, self.less_len,
					self.beam_decoder if self.eval_beam_search else None)[1]
				to_sentences(result['fw']['hard_outputs'], result['fw']['hard_outputs_lens'], self.itos, trans_sens)
				# get_topk(result['fw']['logits'], result['fw']['hard_outputs_lens_with_eos'], prob_list, ind_list, self.para_top_num)

		src_sens = list(dataset_loader.dataset.text)
		src_labels = list(dataset_loader.dataset.label)
		print('src example (index 0 out of {}): {}'.format(len(src_sens), src_sens[0]))

		trans_sens = reorder(dataset_loader.order, trans_sens)
		print('trans example (index 0 out of {}): {}'.format(len(trans_sens), trans_sens[0]))

		# prob_list = reorder(dataset_loader.order, prob_list)
		# ind_list = reorder(dataset_loader.order, ind_list)
		
		result_file = f'{self.pseudo_path}-{self.step-1}'
		print('saving parallel results to file: {}'.format(result_file))
		save_parallel_results(src_sens, src_labels, trans_sens, result_file)

		print('{:.2f} s for parallel data generation'.format(time.time() - start))
		if not self.eval_dropout:
			self.model.train()
		self.train_loaders[0].dataset.fields['text'].postprocessing.to_train_mode()
		
if __name__=='__main__':

	parser=argparse.ArgumentParser()
	parser.add_argument('version', type=str)
	parser.add_argument('-seed', type=int, default=1)
	parser.add_argument('-gpu', type=int, default=0)
	parser.add_argument('-debug', type=str2bool, default=False)
	parser.add_argument('-cudnn_enabled', type=str2bool, default=True)
	parser.add_argument('-save_para_train', type=str2bool, default=False)
	parser.add_argument('-para_top_num', type=int, default=10, nargs='?')
	parser.add_argument('-eval_only', type=str2bool, default=False)
	parser.add_argument('-eval_beam_search', type=str2bool, default=False)
	parser.add_argument('-eval_dropout', type=str2bool, default=False)
	parser.add_argument('-keep_only_full', type=str2bool, default=True)
	parser.add_argument('-label_split', type=str2bool, default=True)
	parser.add_argument('-load_optim', type=str2bool, default=True)
	parser.add_argument('-batch_merge', type=str2bool, default=False)
	parser.add_argument('-opt_accross_styles', type=str2bool, default=False, nargs='?')
	parser.add_argument('-train_from', type=str, default=None, nargs='?')
	parser.add_argument('-style_train_tool', type=str, default=None, nargs='?')
	parser.add_argument('-style_eval_tool', type=str, default=None)
	parser.add_argument('-fluency_train_tool', type=str, default=None, nargs='?')
	parser.add_argument('-fluency_eval_tool', type=str, default=None)
	parser.add_argument('-fluency_eval_tool_rev', type=str, default=None, nargs='?')
	parser.add_argument('-aux_model_eval_mode', type=str2bool, default=False, nargs='?')
	parser.add_argument('-num_ref', type=int, default=1)
	parser.add_argument('-sub', type=float, default=1)
	parser.add_argument('-pretrained_emb', type=str, default=None, nargs='?')
	parser.add_argument('-pretrained_emb_max', type=int, default=None, nargs='?')
	parser.add_argument('-pseudo_path', type=str, default=None, nargs='?')

	parser.add_argument('-work_dir', type=str, default='./')
	parser.add_argument('-dataset', type=str, default='yelp')
	parser.add_argument('-max_sen_len', type=int, default=None, nargs='?')
	parser.add_argument('-min_freq', type=int, default=1)
	parser.add_argument('-max_vocab_size', type=int, default=None, nargs='?')
	parser.add_argument('-noise_drop', type=float, default=0.1, nargs='?')
	parser.add_argument('-noise_drop_as_unk', type=float, default=1.0, nargs='?')
	parser.add_argument('-noise_insert', type=float, default=0.0, nargs='?')
	parser.add_argument('-noise_insert_self', type=float, default=0.0, nargs='?')
	parser.add_argument('-mask_src_rate', type=float, default=0.0)
	parser.add_argument('-mask_src_consc', type=str2bool, default=False, nargs='?')
	parser.add_argument('-mask_src_span', type=str2bool, default=False, nargs='?')
	parser.add_argument('-mask_src_span_len', type=int, default=2, nargs='?')
	parser.add_argument('-batch_size', type=int, default=64)
	parser.add_argument('-eval_batch_size', type=int, default=64)
	
	parser.add_argument('-use_transformer', type=str2bool, default=True)
	parser.add_argument('-use_att', type=str2bool, default=True, nargs='?')
	parser.add_argument('-emb_size', type=int, default=200)
	parser.add_argument('-emb_max_norm', type=float, default=1.0, nargs='?')
	parser.add_argument('-pooling_size', type=int, default=5, nargs='?')
	parser.add_argument('-rnn_type', type=str, default='GRU', nargs='?')
	parser.add_argument('-hid_size', type=int, default=200)
	parser.add_argument('-dec_hid_size', type=int, default=200, nargs='?')
	parser.add_argument('-num_heads', type=int, default=8, nargs='?')
	parser.add_argument('-enc_num_layers', type=int, default=1)
	parser.add_argument('-dec_num_layers', type=int, default=1)
	parser.add_argument('-h_only', type=str2bool, default=True, nargs='?')
	parser.add_argument('-diff_bias', type=str2bool, default=True)
	parser.add_argument('-feed_last_context', type=str2bool, default=True, nargs='?')
	parser.add_argument('-pos_sincode', type=str2bool, default=True, nargs='?')
	parser.add_argument('-token_emb_scale', type=str2bool, default=True, nargs='?')
	parser.add_argument('-dropout_rate', type=float, default=0.0, nargs='?')
	parser.add_argument('-att_dropout_rate', type=float, default=0, nargs='?')
	parser.add_argument('-transformer_norm_bf', type=str2bool, default=False, nargs='?')
	parser.add_argument('-enc_cat_style', type=str2bool, default=False)
	parser.add_argument('-share_pos_emb', type=str2bool, default=True, nargs='?')

	parser.add_argument('-tau', type=float, default=0.5)
	parser.add_argument('-tau_up', type=str2bool, default=None, nargs='?')
	parser.add_argument('-tau_update_start', type=int, default=None, nargs='?')
	parser.add_argument('-tau_update_end', type=int, default=None, nargs='?')
	parser.add_argument('-greedy_train', type=str2bool, default=True)
	parser.add_argument('-min_len', type=int, default=1, nargs='?')
	parser.add_argument('-less_len', type=int, default=None, nargs='?')
	parser.add_argument('-trans_extra_len', type=int, default=0)
	parser.add_argument('-trans_extra_len_eval', type=int, default=5)
	parser.add_argument('-beam_size', type=int, default=5, nargs='?')
	
	parser.add_argument('-rec_weight', type=float, default=1.0)
	parser.add_argument('-pseudo_weight', type=float, default=0.0)
	parser.add_argument('-simul_weight', type=float, default=0.0)
	parser.add_argument('-bt_weight', type=float, default=1.0)
	parser.add_argument('-clf_weight', type=float, default=1.0)
	parser.add_argument('-lm_weight', type=float, default=0.0)
	parser.add_argument('-enc_pred_weight', type=float, default=0.0)
	parser.add_argument('-lm_tgt_mode', type=str, default='hard', nargs='?', choices=['soft', 'soft_detach', 'hard'])
	parser.add_argument('-clf_adv', type=str2bool, default=False, nargs='?')
	parser.add_argument('-clf_adv_mode', type=str, default='ac', nargs='?')
	parser.add_argument('-clf_adv_scale', type=float, default=1.0, nargs='?')
	parser.add_argument('-clf_adv_lr', type=float, default=0.001, nargs='?')
	parser.add_argument('-clf_adv_src_scale', type=float, default=1.0, nargs='?')
	parser.add_argument('-bt_sg', type=str2bool, default=True, nargs='?')
	parser.add_argument('-smooth', type=float, default=0.0)
	
	parser.add_argument('-start_saving', type=int, default=None, nargs='?')
	parser.add_argument('-weight_up_start', type=int, default=None, nargs='?')
	parser.add_argument('-weight_up_end', type=int, default=None, nargs='?')
	parser.add_argument('-weight_down_start', type=int, default=None, nargs='?')
	parser.add_argument('-weight_down_end', type=int, default=None, nargs='?')
	parser.add_argument('-rec_down', type=str2bool, default=False, nargs='?')
	parser.add_argument('-pseudo_down', type=str2bool, default=False, nargs='?')
	parser.add_argument('-simul_down', type=str2bool, default=False, nargs='?')
	
	parser.add_argument('-max_grad_norm', type=float, default=2.0)
	parser.add_argument('-optim_method', type=str, default='adam')
	parser.add_argument('-momentum', type=float, default=None, nargs='?')
	parser.add_argument('-weight_decay', type=float, default=0)
	parser.add_argument('-beta2', type=float, default=0.999, nargs='?')
	parser.add_argument('-lr', type=float, default=0.001)
	parser.add_argument('-update_interval', type=int, default=500, nargs='?')
	parser.add_argument('-up_alpha', type=float, default=None, nargs='?')
	parser.add_argument('-down_alpha', type=float, default=None, nargs='?')
	parser.add_argument('-lr_method', type=str, default=None, nargs='?')
	parser.add_argument('-lr_warmup_steps', type=int, default=0)
	parser.add_argument('-lr_decay_steps', type=int, default=0)
	parser.add_argument('-lr_decay_mode', type=str, default=None, nargs='?')
	parser.add_argument('-lr_min_factor', type=float, default=None, nargs='?')
	parser.add_argument('-lr_init_factor', type=float, default=None, nargs='?')
	parser.add_argument('-lr_decay_rate', type=float, default=None, nargs='?')
	parser.add_argument('-n_iters', type=int, default=100000)
	parser.add_argument('-log_interval', type=int, default=10)
	parser.add_argument('-eval_interval', type=int, default=500)
	parser.add_argument('-save_start', type=int, default=20000)
	parser.add_argument('-save_interval', type=int, default=1000)


	config=parser.parse_args()
	print(' '.join(sys.argv))
	print(config)

	torch.backends.cudnn.enabled=config.cudnn_enabled

	random.seed(config.seed)
	np.random.seed(config.seed+1)
	torch.manual_seed(config.seed+2)
	torch.cuda.manual_seed(config.seed+3)

	print('Start time: ', time.strftime('%X %x %Z'))
	trainer = Solver(config)
	if config.save_para_train:
		trainer.generate_parallel()
	elif config.eval_only:
		trainer.eval('test', trainer.test_loader, True)
	else:
		trainer.train()
	print('Finish time: ', time.strftime('%X %x %Z'))
