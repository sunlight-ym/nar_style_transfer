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
from layers import get_padding_mask_on_size, reverse_seq, cat_zeros_at_start
from nat_models import nat_style_transfer
from classifiers import *
from lms import *
from models import style_transfer_transformer
# from search import SequenceGenerator
from torch.utils.tensorboard import SummaryWriter
import datalib.constants as constants
#torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark=True



class Solver(object):
	"""docstring for Solver"""
	def __init__(self, config):
		super(Solver, self).__init__()
		self.device = torch.device('cuda', config.gpu)
		self.eval_only = config.eval_only
		self.debug = config.debug
		self.cal_inter = config.cal_inter
		self.save_details = config.save_details
		self.keep_only_full = config.keep_only_full
		self.start_saving = config.start_saving
		
		self.batch_merge = config.batch_merge
		self.opt_accross_styles = config.opt_accross_styles
		self.tau = config.tau
		self.tau_up = config.tau_up
		self.tau_update_start = config.tau_update_start
		self.tau_update_end = config.tau_update_end
		self.eval_dropout = config.eval_dropout
		
		self.rec_weight = config.rec_weight
		self.pseudo_weight = config.pseudo_weight
		self.bt_weight = config.bt_weight
		self.clf_weight = config.clf_weight
		self.lm_weight = config.lm_weight
		self.enc_pred_weight = config.enc_pred_weight
		self.simul_weight = config.simul_weight
		self.simul_mask_weight = config.simul_mask_weight
		self.tch_weight = config.tch_weight
		self.mpx_iter_weight = config.mpx_iter_weight
		self.fm_iter_weight = config.fm_iter_weight
		self.px_hard_alpha = config.px_hard_alpha
		self.self_att_scale = config.self_att_scale
		self.enc_att_scale = config.enc_att_scale
		self.clf_adv = config.clf_adv
		self.clf_adv_mode = config.clf_adv_mode
		self.clf_adv_scale = config.clf_adv_scale
		self.clf_adv_src_scale = config.clf_adv_src_scale
		self.fm_clf = config.fm_clf
		self.mpx_clf = config.mpx_clf
		self.fm_clf_adv = config.fm_clf_adv
		self.mpx_clf_adv = config.mpx_clf_adv
		self.fm_lm = config.fm_lm
		self.mpx_lm = config.mpx_lm

		self.fc_tch_weight = config.fc_tch_weight
		self.fc_bt_weight = config.fc_bt_weight
		self.fc_clf = config.fc_clf
		self.fc_clf_adv = config.fc_clf_adv
		self.mpx_bt_weight = config.mpx_bt_weight
		self.mpx_bt_down = config.mpx_bt_down
		self.cons_tau = config.cons_tau
		self.cons_topk = config.cons_topk

		self.ut_weight = config.ut_weight
		self.fm_ut = config.fm_ut
		self.mpx_ut = config.mpx_ut
		self.fc_ut = config.fc_ut
		self.mpx_ut_down = config.mpx_ut_down
		self.ut_offset = config.ut_offset
		


		self.smooth = config.smooth
		self.lm_tgt_mode = config.lm_tgt_mode
		self.tch_sym = config.tch_sym
		self.iter_sym = config.iter_sym
		self.l2_mode = config.l2_mode
		self.att_l2_mode = config.att_l2_mode
		self.att_loss_layers = config.att_loss_layers
		self.enc_cat_style = config.enc_cat_style

		self.dec_mask_mode = config.dec_mask_mode
		self.para_x_fm = config.para_x_fm
		self.fm_recomp_lens = config.fm_recomp_lens 
		self.less_len = config.less_len 
		self.bt_use_recomp_lens = config.bt_use_recomp_lens
		self.dy_extra_len = config.dy_extra_len
		self.extra_len = config.extra_len

		self.need_para = self.rec_weight > 0 or self.pseudo_weight > 0 or self.bt_weight > 0 or self.mpx_bt_weight > 0 or self.fc_bt_weight > 0
		self.need_mpx = self.simul_mask_weight > 0 or self.tch_weight > 0 or self.mpx_iter_weight > 0 or self.mpx_bt_weight > 0 or (
			self.clf_weight > 0 and self.mpx_clf) or (self.clf_adv and self.clf_weight > 0 and self.mpx_clf_adv) or (self.ut_weight > 0 and self.mpx_ut)
		self.need_pseudo = self.pseudo_weight > 0 or self.simul_weight > 0 or self.need_mpx
		self.need_mx_dec = self.need_para and self.dec_mask_mode and not self.para_x_fm
		self.need_pseudo_top = (self.simul_weight > 0 or self.simul_mask_weight > 0) and self.px_hard_alpha < 1

		self.fm_fill_eos = config.fm_fill_eos
		self.len_offset_l = config.len_offset_l
		self.len_offset_r = config.len_offset_r
		self.simple_lp = config.simple_lp
		self.lp_value = config.lp_value
		self.lp_rela = config.lp_rela
		self.lp_cb_rela = config.lp_cb_rela
		self.lp_cb_add = config.lp_cb_add
		self.lp_cb_simple = config.lp_cb_simple
		self.lp_cb_value = config.lp_cb_value
		self.iter_num = config.iter_num
		self.mask_repeat = config.mask_repeat
		self.emit_mi = config.emit_mi
		self.ret_mi = config.ret_mi
		self.mi_alpha = config.mi_alpha
		self.all_iter_eval = config.all_iter_eval
		self.rescore_mode = config.rescore_mode
		self.rescore_beta = config.rescore_beta

		self.add_cov = config.add_cov
		self.cov_mode = config.cov_mode
		self.cov_weight = config.cov_weight
		self.cov_inv = config.cov_inv
		self.cov_with_start = config.cov_with_start
		self.ctr_use_t1 = config.ctr_use_t1
		self.ctr_use_t2 = config.ctr_use_t2
		self.ctr_by_seq = config.ctr_by_seq
		self.ctr_margin = config.ctr_margin
		self.ctr_fc_margin = config.ctr_fc_margin
		self.ctr_weight = config.ctr_weight
		self.ctr_guide_scale = config.ctr_guide_scale
		self.ctr_delete = config.ctr_delete
		self.ctr_insert = config.ctr_insert
		self.ctr_guide_delete = config.ctr_guide_delete
		self.ctr_guide_insert = config.ctr_guide_insert
		self.ctr_start = config.ctr_start
		self.ctr_n = config.ctr_n
		self.rec_ctr = config.rec_ctr
		self.px_ctr = config.px_ctr
		self.mpx_ctr = config.mpx_ctr
		self.mpx_bt_ctr = config.mpx_bt_ctr
		self.fc_bt_ctr = config.fc_bt_ctr
		self.fc_ctr = config.fc_ctr
		self.fc_ctr_guide = config.fc_ctr_guide

		if self.rescore_mode == 'at':
			check_point = torch.load(config.rescore_at_model, map_location=lambda storage, loc: storage)
			self.rescore_at_model = style_transfer_transformer(*check_point['args'])
			self.rescore_at_model.to(self.device)
			self.rescore_at_model.load_state_dict(check_point['model_state'])
			del check_point
			# self.style_eval_tool = torch.load(config.style_eval_tool, map_location=self.device)['model']
			frozen_model(self.rescore_at_model)
			self.rescore_at_model.eval()

		
		self.weight_up_start = config.weight_up_start
		self.weight_up_end = config.weight_up_end
		self.pre_weight_up_start = config.pre_weight_up_start
		self.pre_weight_up_end = config.pre_weight_up_end
		self.weight_down_start = config.weight_down_start
		self.weight_down_end = config.weight_down_end
		self.rec_down = config.rec_down
		self.pseudo_down = config.pseudo_down
		self.simul_down = config.simul_down
		self.simul_mask_down = config.simul_mask_down
		self.tch_down = config.tch_down
		self.mpx_iter_down = config.mpx_iter_down
		self.mpx_clf_down = config.mpx_clf_down
		self.mpx_clf_adv_down = config.mpx_clf_adv_down
		self.mpx_lm_down = config.mpx_lm_down

		self.max_grad_norm = config.max_grad_norm
		self.update_interval = config.update_interval
		self.up_alpha = config.up_alpha
		self.down_alpha = config.down_alpha
		self.n_iters = config.n_iters
		self.log_interval = config.log_interval
		self.eval_interval = config.eval_interval
		self.aux_model_eval_mode = config.aux_model_eval_mode
		self.save_start = config.save_start
		self.save_interval = config.save_interval
		

		self.static_arg_dict = {'greedy':config.greedy_train, 'fm_fill_eos':config.fm_fill_eos, 
				'enc_use_x':config.mask_src_rate==0, 'enc_pred':self.enc_pred_weight>0, 
				'tch_keep_rate':config.tch_keep_rate, 'iter_keep_rate':config.iter_keep_rate,
				'iter_random':config.iter_random, 'fm_use_pseudo_len':config.fm_use_pseudo_len and self.need_pseudo, 
				'fm_recomp_lens':config.fm_recomp_lens,'less_len':config.less_len, 'bt_sg':config.bt_sg, 
				'bt_use_recomp_lens':config.bt_use_recomp_lens, 'dec_mask_mode':config.dec_mask_mode,
				'para_x_fm':config.para_x_fm, 'tch_sym':config.tch_sym, 'iter_sym':config.iter_sym,
				'need_self_attn':config.self_att_scale>0, 'need_enc_attn':config.enc_att_scale>0,
				'fc_iter_num':config.fc_iter_num, 'fc_mask_rate':config.fc_mask_rate,
				'fc_mask_mode':config.fc_mask_mode, 'fc_mask_largest':config.fc_mask_largest,
				'ctr_word_im':config.ctr_word_im, 'ctr_n':config.ctr_n, 'ctr_kmin':config.ctr_kmin,
				'ctr_kmax':config.ctr_kmax, 'ctr_fc_good_cov':config.ctr_fc_good_cov, 'ctr_fc_bad_cov':config.ctr_fc_bad_cov,
				'cov_mode':config.cov_mode, 'cov_with_start':config.cov_with_start}

		

		self.model_path = os.path.join(config.work_dir, 'model', config.dataset, 'nattrans', config.version)
		makedirs(self.model_path)
		self.output_path = os.path.join(config.work_dir, 'output', config.dataset, 'nattrans', config.version)
		makedirs(self.output_path)
		if not self.eval_only:
			self.summary_path = os.path.join(config.work_dir, 'summary', config.dataset, 'nattrans', config.version)
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
							x_mask_tgt=self.need_mx_dec, mask_tgt_consc=config.mask_tgt_consc, 
							mask_tgt_span=config.mask_tgt_span, mask_tgt_span_len=config.mask_tgt_span_len, 
							px_mask_target=self.need_mpx, require_px_top=self.need_pseudo_top,
							with_pseudo=self.need_pseudo, pseudo_path=config.pseudo_path)
		if config.sub < 1:
			old_len = len(datasets[0])
			random.shuffle(datasets[0].examples)
			new_len = int(old_len * config.sub)
			datasets[0].examples = datasets[0].examples[:new_len]
			print('cutting the training data from {} to {}'.format(old_len, len(datasets[0])))
		self.valid_loader, self.test_loader = BucketIterator.splits(datasets[1:], 
							batch_size = config.eval_batch_size, 
							train_flags = [False]*2)
		

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
		
		
		self.args = [vocab_size, config.emb_size, config.emb_max_norm, config.pos_sincode, config.token_emb_scale, config.max_sen_len,
			config.dropout_rate, config.num_heads, config.hid_size, config.enc_num_layers, config.dec_num_layers, config.diff_bias, num_classes,
			config.att_dropout_rate, config.transformer_norm_bf, config.enc_cat_style, config.share_pos_emb,
			config.positional_att, config.apply_self_mask, config.self_mask_escape_start]
		self.model = nat_style_transfer(vocab_size, config.emb_size, config.emb_max_norm, config.pos_sincode, config.token_emb_scale, config.max_sen_len,
			config.dropout_rate, config.num_heads, config.hid_size, config.enc_num_layers, config.dec_num_layers, config.diff_bias, num_classes,
			config.att_dropout_rate, config.transformer_norm_bf, config.enc_cat_style, config.share_pos_emb,
			config.positional_att, config.apply_self_mask, config.self_mask_escape_start)
		
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
		if config.train_from is not None or config.train_from_at is not None:
			middle_training = config.train_from is not None
			pre_path = config.train_from if middle_training else config.train_from_at
			check_point=torch.load(pre_path, map_location=lambda storage, loc: storage)
			load_msg = self.model.load_state_dict(check_point['model_state'], strict=middle_training)
			print(load_msg)
			if self.clf_weight > 0 and self.clf_adv and ('clf_model_state' in check_point):
				self.style_train_tool_update.load_state_dict(check_point['clf_model_state'])
				self.style_train_tool.load_state_dict(check_point['clf_model_state'])
			if middle_training:
				if config.load_optim:
					self.optimizer.load_state_dict(check_point['optimizer_state'])
					self.lr_scheduler.load_state_dict(check_point['lr_scheduler_state'])
					if self.clf_weight > 0 and self.clf_adv and ('clf_model_state' in check_point):
						self.clf_adv_optimizer.load_state_dict(check_point['clf_adv_optimizer_state'])
					
				self.step = check_point['step'] + 1
			del check_point

		
		

	def save_states(self, prefix = ''):
		check_point = {
			'args': self.args,
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
		
		xc, clens, x, lens, mx_enc_in, mx_enc_out, mx_dec_in, mx_dec_out = [
			batch.text.get(k) for k in [
			'xc', 'cenc_lens', 'x', 'enc_lens', 'mx_enc_in', 'mx_enc_out', 'mx_dec_in', 'mx_dec_out']]
		style = batch.label
		mx_enc_mask = (mx_enc_out != constants.PAD_ID) if mx_enc_in is not None and self.enc_pred_weight > 0 else None
		mx_dec_mask = (mx_dec_out != constants.PAD_ID) if mx_dec_in is not None else None
		cseq_mask_b = get_padding_mask_on_size(xc.size(0), clens, 0, 1) if xc is not None else None
		seq_mask = get_padding_mask_on_size(x.size(0), lens)
		seq_mask_b = seq_mask.t().contiguous()
		batch_dict = {'style':style, 'xc':xc, 'clens':clens, 'cseq_mask_b':cseq_mask_b, 
				'x':x, 'lens':lens, 'seq_mask':seq_mask, 'seq_mask_b':seq_mask_b, 
				'mx_enc_in':mx_enc_in, 'mx_enc_mask':mx_enc_mask, 
				'mx_dec_in':mx_dec_in, 'mx_dec_mask':mx_dec_mask, 
				}
		if hasattr(batch, 'pseudo'):
			px, plens, mpx_dec_in, mpx_dec_out = [batch.pseudo.get(k) for k in ['x', 'enc_lens', 'mx_dec_in', 'mx_dec_out']]
			mpx_dec_mask = (mpx_dec_out != constants.PAD_ID) if mpx_dec_in is not None else None
			pseq_mask_b = get_padding_mask_on_size(px.size(0), plens, 0, 1)
			pseq_mask = pseq_mask_b.t().contiguous()
			batch_dict.update({'px':px, 'plens':plens, 'pseq_mask_b':pseq_mask_b, 'pseq_mask':pseq_mask,
				'mpx_dec_in':mpx_dec_in, 'mpx_dec_mask':mpx_dec_mask})
		if hasattr(batch, 'topp'):
			batch_dict.update({'px_topp':batch.topp, 'px_topi':batch.topi})
		
		return batch_dict
	def prepare_input_for_clf(self, clf_model, result, soft, hard_outputs_name='hard_outputs', padding_name='hard_outputs_padding_mask_with_eos',
		padding_b_name='hard_outputs_padding_mask_t_with_eos', len_name='hard_outputs_lens_with_eos'):
		batch_dict_for_clf = {}
		if isinstance(clf_model, cnn_classifier):
			batch_dict_for_clf['x_b'] = result['soft_outputs' if soft else hard_outputs_name].transpose(0, 1).contiguous()
			batch_dict_for_clf['padding_mask_b'] = result[padding_b_name]
			batch_dict_for_clf['x_b'] = batch_dict_for_clf['x_b'].masked_fill(batch_dict_for_clf['padding_mask_b'].unsqueeze(-1) 
				if soft else batch_dict_for_clf['padding_mask_b'], 0 if soft else constants.PAD_ID)
		elif isinstance(clf_model, attn_classifier):
			batch_dict_for_clf['x'] = result['soft_outputs' if soft else hard_outputs_name]
			batch_dict_for_clf['padding_mask'] = result[padding_name]
		else:
			batch_dict_for_clf['x'] = result['soft_outputs' if soft else hard_outputs_name]
			batch_dict_for_clf['padding_mask_b'] = result[padding_b_name]
			batch_dict_for_clf['lens'] = result[len_name]
		return batch_dict_for_clf
	def combine(self, result, org_input, mask, soft_out):
		if soft_out:
			pred_soft = result['soft_outputs']
			mpx_soft = pred_soft.new_zeros(pred_soft.size()).scatter_(-1, org_input.unsqueeze(-1), 1)
			combined_input = mpx_soft.masked_scatter(mask.unsqueeze(-1), pred_soft[mask])
		else:
			pred_hard = result['hard_outputs']
			combined_input = org_input.masked_scatter(mask, pred_hard[mask])
		return combined_input

	def prepare_input_for_clf_from_mx(self, clf_model, result, batch_dict, soft):
		batch_dict_for_clf = {}
		combined_input = self.combine(result, batch_dict['px'], batch_dict['mpx_dec_mask'], soft)
		if isinstance(clf_model, cnn_classifier):
			batch_dict_for_clf['x_b'] = combined_input.transpose(0, 1).contiguous()
			batch_dict_for_clf['padding_mask_b'] = batch_dict['pseq_mask_b']
		elif isinstance(clf_model, attn_classifier):
			batch_dict_for_clf['x'] = combined_input
			batch_dict_for_clf['padding_mask'] = batch_dict['pseq_mask']
		else:
			batch_dict_for_clf['x'] = combined_input
			batch_dict_for_clf['padding_mask_b'] = batch_dict['pseq_mask_b']
			batch_dict_for_clf['lens'] = batch_dict['plens']
		return batch_dict_for_clf
	def prepare_input_for_clf_from_fc(self, clf_model, result, soft):
		batch_dict_for_clf = {}
		combined_input = self.combine(result, result['fc_hard_target'], result['fc_mask'], soft)
		if isinstance(clf_model, cnn_classifier):
			batch_dict_for_clf['x_b'] = combined_input.transpose(0, 1).contiguous()
			batch_dict_for_clf['padding_mask_b'] = result['fc_padding_mask_b']
			batch_dict_for_clf['x_b'] = batch_dict_for_clf['x_b'].masked_fill(batch_dict_for_clf['padding_mask_b'].unsqueeze(-1) 
				if soft else batch_dict_for_clf['padding_mask_b'], 0 if soft else constants.PAD_ID)
		elif isinstance(clf_model, attn_classifier):
			batch_dict_for_clf['x'] = combined_input
			batch_dict_for_clf['padding_mask'] = result['fc_padding_mask']
		else:
			batch_dict_for_clf['x'] = combined_input
			batch_dict_for_clf['padding_mask_b'] = result['fc_padding_mask_b']
			batch_dict_for_clf['lens'] = result['fc_lens']
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
			batch_dict_for_clf['lens'] = batch_dict['lens']
		return batch_dict_for_clf

	def compute_loss(self, result, batch_dict, arg_dict, size_average):
		loss_values = OrderedDict()
		x, lens, seq_mask, style, mx_enc_mask, mx_dec_mask, px, plens, pseq_mask, pseq_mask_b, mpx_dec_mask, px_topp, px_topi, seq_mask_b = [
			batch_dict.get(k) for k in ['x', 'lens', 'seq_mask', 'style', 'mx_enc_mask', 'mx_dec_mask',
			'px', 'plens', 'pseq_mask', 'pseq_mask_b', 'mpx_dec_mask', 'px_topp', 'px_topi', 'seq_mask_b']]

		if ('rec' in result or 'pseudo' in result or 'bt' in result or 'mpx_bt' in result or 'fc_bt' in result) and not self.need_mx_dec:
			if self.dec_mask_mode and arg_dict['fm_fill_eos']:
				para_lens = lens - 1
				para_seq_mask = seq_mask | (x == constants.EOS_ID)
			else:
				para_lens = lens
				para_seq_mask = seq_mask


		loss_all = 0
		down_weight = arg_dict['down_weight']
		up_weight = arg_dict['up_weight']
		pre_up_weight = arg_dict['pre_up_weight']
		if 'rec' in result:
			
			scale = (down_weight if self.rec_down else 1) * self.rec_weight
			rec_loss = (masked_prediction_loss(result['rec']['logits'], x, mx_dec_mask, size_average, smooth=self.smooth) 
				if self.need_mx_dec 
				else seq_ce_logits_loss(result['rec']['logits'], x, para_lens, para_seq_mask, size_average, smooth=self.smooth))
			loss_values['rec'] = rec_loss.item()
			loss_all = loss_all + scale * rec_loss
			loss_values['rec_acc'] = (masked_prediction_acc(result['rec']['logits'], x, mx_dec_mask, size_average) 
				if self.need_mx_dec 
				else seq_acc(result['rec']['logits'], x, para_lens, para_seq_mask, size_average))
			if 'rec_dctr' in result:
				if self.ctr_guide_scale > 0 and self.ctr_guide_delete:
					rec_dctr_guide_loss = 0
					for i in range(self.ctr_n):
						rec_dctr_guide_loss = rec_dctr_guide_loss + (
							masked_prediction_loss(result['rec_dctr'][i], x, result['rec_dctr_masks'][i], size_average, smooth=self.smooth) 
							if self.need_mx_dec 
							else seq_ce_logits_loss(result['rec_dctr'][i], x, 
							(~result['rec_dctr_masks'][i]).long().sum(0), result['rec_dctr_masks'][i], size_average, smooth=self.smooth))
					loss_values['rec_dctr_guide'] = rec_dctr_guide_loss.item()
					loss_all = loss_all + scale * self.ctr_guide_scale * rec_dctr_guide_loss
				rec_dctr_loss = (mx_contrast_loss(result['rec']['logits'], result['rec_dctr'], 
									mx_dec_mask, result['rec_dctr_masks'], x, self.ctr_use_t1, self.ctr_use_t2, 
									self.ctr_margin, self.ctr_by_seq, result['rec_dctr_word_ims'], [None]*self.ctr_n, 
									result['rec_dctr_unchange'], size_average)
									if self.need_mx_dec else 
									seq_contrast_loss(result['rec']['logits'], result['rec_dctr'], 
									seq_mask, result['rec_dctr_masks'], x, self.ctr_use_t1, self.ctr_use_t2, 
									self.ctr_margin, self.ctr_by_seq, result['rec_dctr_word_ims'], [None]*self.ctr_n, 
									result['rec_dctr_unchange'], size_average))
				loss_values['rec_dctr'] = rec_dctr_loss.item()
				loss_all = loss_all + self.ctr_weight * rec_dctr_loss
			if 'rec_ictr' in result:
				if self.ctr_guide_scale > 0 and self.ctr_guide_insert:
					rec_ictr_guide_loss = 0
					for i in range(self.ctr_n):
						x_reform = reform_target(x, result['rec_ictr_filling_masks'][i])
						rec_ictr_guide_loss = rec_ictr_guide_loss + (
							masked_prediction_loss_reform(result['rec_ictr'][i], x_reform, 
							result['rec_ictr_masks'][i], reform_mm(result['rec_ictr_masks'][i], result['rec_ictr_filling_masks'][i]), 
							size_average, smooth=self.smooth) 
							if self.need_mx_dec 
							else seq_ce_logits_loss(result['rec_ictr'][i], x_reform, 
							lens, reform_pm(result['rec_ictr_masks'][i], result['rec_ictr_filling_masks'][i]), 
							size_average, smooth=self.smooth))
					loss_values['rec_ictr_guide'] = rec_ictr_guide_loss.item()
					loss_all = loss_all + scale * self.ctr_guide_scale * rec_ictr_guide_loss
				rec_ictr_loss = (mx_contrast_insert_loss(result['rec']['logits'], result['rec_ictr'], 
									mx_dec_mask, result['rec_ictr_masks'], x, self.ctr_use_t1, 
									self.ctr_margin, self.ctr_by_seq, result['rec_ictr_word_ims'], [None]*self.ctr_n, 
									result['rec_ictr_filling_masks'], size_average)
									if self.need_mx_dec else 
									seq_contrast_insert_loss(result['rec']['logits'], result['rec_ictr'], 
									seq_mask, result['rec_ictr_masks'], x, self.ctr_use_t1, 
									self.ctr_margin, self.ctr_by_seq, result['rec_ictr_word_ims'], [None]*self.ctr_n, 
									result['rec_ictr_filling_masks'], size_average))
				loss_values['rec_ictr'] = rec_ictr_loss.item()
				loss_all = loss_all + self.ctr_weight * rec_ictr_loss
		if 'pseudo' in result:
			scale = (down_weight if self.pseudo_down else 1) * self.pseudo_weight
			pseudo_loss = (masked_prediction_loss(result['pseudo']['logits'], x, mx_dec_mask, size_average, smooth=self.smooth) 
				if self.need_mx_dec 
				else seq_ce_logits_loss(result['pseudo']['logits'], x, para_lens, para_seq_mask, size_average, smooth=self.smooth))
			loss_values['pseudo'] = pseudo_loss.item()
			loss_all = loss_all + scale * pseudo_loss
			loss_values['pseudo_acc'] = (masked_prediction_acc(result['pseudo']['logits'], x, mx_dec_mask, size_average) 
				if self.need_mx_dec 
				else seq_acc(result['pseudo']['logits'], x, para_lens, para_seq_mask, size_average))
		
		if arg_dict['enc_pred']:
			scale = self.enc_pred_weight
			enc_pred_loss = masked_prediction_loss(result['enc_logits'], x, mx_enc_mask, size_average)
			loss_values['enc_pred'] = enc_pred_loss.item()
			loss_values['enc_pred_acc'] = masked_prediction_acc(result['enc_logits'], x, mx_enc_mask, size_average)
			loss_all = loss_all + scale * enc_pred_loss
		
		if 'dec_px' in result:
			if self.dec_mask_mode and arg_dict['fm_fill_eos']:
				px_plens = plens - 1
				px_pseq_mask = pseq_mask | (px == constants.EOS_ID)
			else:
				px_plens = plens
				px_pseq_mask = pseq_mask
			scale = (down_weight if self.simul_down else 1) * self.simul_weight
			if self.px_hard_alpha > 0:
				simul_hard_loss = seq_ce_logits_loss(result['dec_px']['logits'], px, px_plens, px_pseq_mask, size_average, smooth=self.smooth)
				loss_values['simul_hard'] = simul_hard_loss.item()
				loss_all = loss_all + scale * self.px_hard_alpha * simul_hard_loss
				loss_values['simul_hard_acc'] = seq_acc(result['dec_px']['logits'], px, px_plens, px_pseq_mask, size_average)
			if self.px_hard_alpha < 1:
				simul_soft_loss = seq_ce_logits_topk_loss(result['dec_px']['logits'], px_topi, px_topp, px_plens, px_pseq_mask, size_average, smooth=self.smooth)
				loss_values['simul_soft'] = simul_soft_loss.item()
				loss_all = loss_all + scale * (1 - self.px_hard_alpha) * simul_soft_loss
			if 'px_dctr' in result:
				if self.ctr_guide_scale > 0 and self.ctr_guide_delete:
					px_dctr_guide_loss = 0
					for i in range(self.ctr_n):
						px_dctr_guide_loss = px_dctr_guide_loss + seq_ce_logits_loss(result['px_dctr'][i], px, (~result['px_dctr_masks'][i]).long().sum(0), 
							result['px_dctr_masks'][i], size_average, smooth=self.smooth)
					loss_values['px_dctr_guide'] = px_dctr_guide_loss.item()
					loss_all = loss_all + scale * self.ctr_guide_scale * px_dctr_guide_loss
				px_dctr_loss = seq_contrast_loss(result['dec_px']['logits'], result['px_dctr'], 
									pseq_mask, result['px_dctr_masks'], px, self.ctr_use_t1, self.ctr_use_t2, 
									self.ctr_margin, self.ctr_by_seq, result['px_dctr_word_ims'], [None]*self.ctr_n, 
									result['px_dctr_unchange'], size_average)
				loss_values['px_dctr'] = px_dctr_loss.item()
				loss_all = loss_all + self.ctr_weight * px_dctr_loss
			if 'px_ictr' in result:
				if self.ctr_guide_scale > 0 and self.ctr_guide_insert:
					px_ictr_guide_loss = 0
					for i in range(self.ctr_n):
						px_ictr_guide_loss = px_ictr_guide_loss + seq_ce_logits_loss(result['px_ictr'][i], 
							reform_target(px, result['px_ictr_filling_masks'][i]), plens, 
							reform_pm(result['px_ictr_masks'][i], result['px_ictr_filling_masks'][i]), size_average, smooth=self.smooth)
					loss_values['px_ictr_guide'] = px_ictr_guide_loss.item()
					loss_all = loss_all + scale * self.ctr_guide_scale * px_ictr_guide_loss
				px_ictr_loss = seq_contrast_insert_loss(result['dec_px']['logits'], result['px_ictr'], 
									pseq_mask, result['px_ictr_masks'], px, self.ctr_use_t1, 
									self.ctr_margin, self.ctr_by_seq, result['px_ictr_word_ims'], [None]*self.ctr_n, 
									result['px_ictr_filling_masks'], size_average)
				loss_values['px_ictr'] = px_ictr_loss.item()
				loss_all = loss_all + self.ctr_weight * px_ictr_loss


		if 'dec_mpx' in result:
			scale = (down_weight if self.simul_mask_down else 1) * self.simul_mask_weight
			mpx_lens = mpx_dec_mask.long().sum(0)
			mpx_seq_mask = ~mpx_dec_mask
			if self.simul_mask_weight > 0 and not (self.simul_mask_down and arg_dict['down_stage_over']):
				if self.px_hard_alpha > 0:
					simul_mask_loss = seq_ce_logits_loss(result['dec_mpx']['logits'], px, mpx_lens, mpx_seq_mask, size_average, smooth=self.smooth)
					loss_values['simul_mask'] = simul_mask_loss.item()
					loss_all = loss_all + scale * self.px_hard_alpha * simul_mask_loss
					loss_values['simul_mask_acc'] = seq_acc(result['dec_mpx']['logits'], px, mpx_lens, mpx_seq_mask, size_average)
				if self.px_hard_alpha < 1:
					simul_mask_soft_loss = seq_ce_logits_topk_loss(result['dec_mpx']['logits'], px_topi, px_topp, mpx_lens, mpx_seq_mask, size_average, smooth=self.smooth)
					loss_values['simul_mask_soft'] = simul_mask_soft_loss.item()
					loss_all = loss_all + scale * (1 - self.px_hard_alpha) * simul_mask_soft_loss
			if 'mpx_dctr' in result:
				mpx_dctr_seq_masks = [~m for m in result['mpx_dctr_masks']]
				if self.ctr_guide_scale > 0 and self.ctr_guide_delete:
					mpx_dctr_guide_loss = 0
					
					for i in range(self.ctr_n):
						mpx_dctr_guide_loss = mpx_dctr_guide_loss + seq_ce_logits_loss(result['mpx_dctr'][i], px, result['mpx_dctr_masks'][i].long().sum(0), 
							mpx_dctr_seq_masks[i], size_average, smooth=self.smooth)
					loss_values['mpx_dctr_guide'] = mpx_dctr_guide_loss.item()
					loss_all = loss_all + scale * self.ctr_guide_scale * mpx_dctr_guide_loss
				mpx_dctr_loss = seq_contrast_loss(result['dec_mpx']['logits'], result['mpx_dctr'], 
									mpx_seq_mask, mpx_dctr_seq_masks, px, self.ctr_use_t1, self.ctr_use_t2, 
									self.ctr_margin, self.ctr_by_seq, result['mpx_dctr_word_ims'], [None]*self.ctr_n,
									result['mpx_dctr_unchange'], size_average)
				loss_values['mpx_dctr'] = mpx_dctr_loss.item()
				loss_all = loss_all + self.ctr_weight * mpx_dctr_loss
			if 'mpx_ictr' in result:
				mpx_ictr_seq_masks = [~m for m in result['mpx_ictr_masks']]
				if self.ctr_guide_scale > 0 and self.ctr_guide_insert:
					mpx_ictr_guide_loss = 0
					
					for i in range(self.ctr_n):
						mpx_ictr_guide_loss = mpx_ictr_guide_loss + seq_ce_logits_loss(result['mpx_ictr'][i], 
							reform_target(px, result['mpx_ictr_filling_masks'][i]), mpx_dec_mask.long().sum(0), 
							reform_pm(mpx_ictr_seq_masks[i], result['mpx_ictr_filling_masks'][i]), size_average, smooth=self.smooth)
					loss_values['mpx_ictr_guide'] = mpx_ictr_guide_loss.item()
					loss_all = loss_all + scale * self.ctr_guide_scale * mpx_ictr_guide_loss
				mpx_ictr_loss = seq_contrast_insert_loss(result['dec_mpx']['logits'], result['mpx_ictr'], 
									mpx_seq_mask, mpx_ictr_seq_masks, px, self.ctr_use_t1, 
									self.ctr_margin, self.ctr_by_seq, result['mpx_ictr_word_ims'], [None]*self.ctr_n,
									result['mpx_ictr_filling_masks'], size_average)
				loss_values['mpx_ictr'] = mpx_ictr_loss.item()
				loss_all = loss_all + self.ctr_weight * mpx_ictr_loss
		if 'mpx_bt' in result:
			scale = (down_weight if self.mpx_bt_down else 1) * self.mpx_bt_weight * pre_up_weight
			mpx_bt_loss = (masked_prediction_loss(result['mpx_bt']['logits'], x, mx_dec_mask, size_average, smooth=self.smooth) 
				if self.need_mx_dec 
				else seq_ce_logits_loss(result['mpx_bt']['logits'], x, para_lens, para_seq_mask, size_average, smooth=self.smooth))
			loss_values['mpx_bt'] = mpx_bt_loss.item()
			loss_all = loss_all + scale * mpx_bt_loss
			loss_values['mpx_bt_acc'] = (masked_prediction_acc(result['mpx_bt']['logits'], x, mx_dec_mask, size_average) 
				if self.need_mx_dec 
				else seq_acc(result['mpx_bt']['logits'], x, para_lens, para_seq_mask, size_average))
			if 'mpx_bt_dctr' in result:
				if self.ctr_guide_scale > 0 and self.ctr_guide_delete:
					mpx_bt_dctr_guide_loss = 0
					for i in range(self.ctr_n):
						mpx_bt_dctr_guide_loss = mpx_bt_dctr_guide_loss + (masked_prediction_loss(result['mpx_bt_dctr'][i], x, result['mpx_bt_dctr_masks'][i], size_average, smooth=self.smooth) 
										if self.need_mx_dec else 
										seq_ce_logits_loss(result['mpx_bt_dctr'][i], x, 
										(~result['mpx_bt_dctr_masks'][i]).long().sum(0), result['mpx_bt_dctr_masks'][i], size_average, smooth=self.smooth))
					loss_values['mpx_bt_dctr_guide'] = mpx_bt_dctr_guide_loss.item()
					loss_all = loss_all + scale * self.ctr_guide_scale * mpx_bt_dctr_guide_loss
				mpx_bt_dctr_loss = (mx_contrast_loss(result['mpx_bt']['logits'], result['mpx_bt_dctr'], 
									mx_dec_mask, result['mpx_bt_dctr_masks'], x, self.ctr_use_t1, self.ctr_use_t2, 
									self.ctr_margin, self.ctr_by_seq, result['mpx_bt_dctr_word_ims'], [None]*self.ctr_n, 
									result['mpx_bt_dctr_unchange'], size_average)
									if self.need_mx_dec else 
									seq_contrast_loss(result['mpx_bt']['logits'], result['mpx_bt_dctr'], 
									seq_mask, result['mpx_bt_dctr_masks'], x, self.ctr_use_t1, self.ctr_use_t2, 
									self.ctr_margin, self.ctr_by_seq, result['mpx_bt_dctr_word_ims'], [None]*self.ctr_n, 
									result['mpx_bt_dctr_unchange'], size_average))
				loss_values['mpx_bt_dctr'] = mpx_bt_dctr_loss.item()
				loss_all = loss_all + self.ctr_weight * mpx_bt_dctr_loss
			if 'mpx_bt_ictr' in result:
				if self.ctr_guide_scale > 0 and self.ctr_guide_insert:
					mpx_bt_ictr_guide_loss = 0
					for i in range(self.ctr_n):
						x_reform = reform_target(x, result['mpx_bt_ictr_filling_masks'][i])
						mpx_bt_ictr_guide_loss = mpx_bt_ictr_guide_loss + (masked_prediction_loss_reform(result['mpx_bt_ictr'][i], x_reform, 
							result['mpx_bt_ictr_masks'][i], reform_mm(result['mpx_bt_ictr_masks'][i], result['mpx_bt_ictr_filling_masks'][i]), 
							size_average, smooth=self.smooth) 
							if self.need_mx_dec else 
							seq_ce_logits_loss(result['mpx_bt_ictr'][i], x_reform, 
							lens, reform_pm(result['mpx_bt_ictr_masks'][i], result['mpx_bt_ictr_filling_masks'][i]), 
							size_average, smooth=self.smooth))
					loss_values['mpx_bt_ictr_guide'] = mpx_bt_ictr_guide_loss.item()
					loss_all = loss_all + scale * self.ctr_guide_scale * mpx_bt_ictr_guide_loss
				mpx_bt_ictr_loss = (mx_contrast_insert_loss(result['mpx_bt']['logits'], result['mpx_bt_ictr'], 
									mx_dec_mask, result['mpx_bt_ictr_masks'], x, self.ctr_use_t1, 
									self.ctr_margin, self.ctr_by_seq, result['mpx_bt_ictr_word_ims'], [None]*self.ctr_n, 
									result['mpx_bt_ictr_filling_masks'], size_average)
									if self.need_mx_dec else 
									seq_contrast_insert_loss(result['mpx_bt']['logits'], result['mpx_bt_ictr'], 
									seq_mask, result['mpx_bt_ictr_masks'], x, self.ctr_use_t1, 
									self.ctr_margin, self.ctr_by_seq, result['mpx_bt_ictr_word_ims'], [None]*self.ctr_n, 
									result['mpx_bt_ictr_filling_masks'], size_average))
				loss_values['mpx_bt_ictr'] = mpx_bt_ictr_loss.item()
				loss_all = loss_all + self.ctr_weight * mpx_bt_ictr_loss
		
		if 'dec_mpx_tch' in result:
			scale = (down_weight if self.tch_down else 1) * self.tch_weight * pre_up_weight
			tch_loss = consistency_loss(result['dec_mpx']['logits'], result['dec_mpx_tch']['logits'], result['mpx_dec_mask_tch'], 
				self.tch_sym, self.l2_mode, size_average, self.cons_tau, self.cons_topk)
			loss_values['tch'] = tch_loss.item()
			loss_all = loss_all + scale * tch_loss
			loss_values['tch_acc'] = consistency_acc(result['dec_mpx']['logits'], result['dec_mpx_tch']['logits'], result['mpx_dec_mask_tch'], size_average)
			if self.self_att_scale > 0:
				plens_self_att = plens + 1
				pseq_mask_b_self_att = cat_zeros_at_start(pseq_mask_b)
				tch_self_att_loss = attn_loss(result['dec_mpx']['attn_weights'], result['dec_mpx_tch']['attn_weights'], 'self_attn_weights', 
					plens_self_att, plens_self_att, pseq_mask_b_self_att, pseq_mask_b_self_att, self.att_loss_layers, self.tch_sym, self.att_l2_mode, size_average)
				loss_values['tch_self_att'] = tch_self_att_loss.item()
				loss_all = loss_all + scale * self.self_att_scale * tch_self_att_loss
			if self.enc_att_scale > 0:
				# lens_enc_att = lens+1 if self.enc_cat_style else lens
				# seq_mask_b_enc_att = cat_zeros_at_start(seq_mask_b) if self.enc_cat_style else seq_mask_b
				plens_enc_att = plens + 1
				pseq_mask_b_enc_att = cat_zeros_at_start(pseq_mask_b)
				tch_enc_att_loss = attn_loss(result['dec_mpx']['attn_weights'], result['dec_mpx_tch']['attn_weights'], 'enc_attn_weights', 
					lens, plens_enc_att, seq_mask_b, pseq_mask_b_enc_att, self.att_loss_layers, self.tch_sym, self.att_l2_mode, size_average)
				loss_values['tch_enc_att'] = tch_enc_att_loss.item()
				loss_all = loss_all + scale * self.enc_att_scale * tch_enc_att_loss
		
		if 'dec_mpx_iter' in result:
			scale = (down_weight if self.mpx_iter_down else 1) * self.mpx_iter_weight * pre_up_weight
			mpx_iter_loss = consistency_loss(result['dec_mpx']['logits'], result['dec_mpx_iter']['logits'], result['mpx_dec_mask_iter'], 
				self.iter_sym, self.l2_mode, size_average, self.cons_tau, self.cons_topk)
			loss_values['mpx_iter'] = mpx_iter_loss.item()
			loss_all = loss_all + scale * mpx_iter_loss
			loss_values['mpx_iter_acc'] = consistency_acc(result['dec_mpx']['logits'], result['dec_mpx_iter']['logits'], result['mpx_dec_mask_iter'], size_average)
			if self.self_att_scale > 0:
				plens_self_att = plens + 1
				pseq_mask_b_self_att = cat_zeros_at_start(pseq_mask_b)
				mpx_iter_self_att_loss = attn_loss(result['dec_mpx']['attn_weights'], result['dec_mpx_iter']['attn_weights'], 'self_attn_weights', 
					plens_self_att, plens_self_att, pseq_mask_b_self_att, pseq_mask_b_self_att, self.att_loss_layers, self.iter_sym, self.att_l2_mode, size_average)
				loss_values['mpx_iter_self_att'] = mpx_iter_self_att_loss.item()
				loss_all = loss_all + scale * self.self_att_scale * mpx_iter_self_att_loss
			if self.enc_att_scale > 0:
				# lens_enc_att = lens+1 if self.enc_cat_style else lens
				# seq_mask_b_enc_att = cat_zeros_at_start(seq_mask_b) if self.enc_cat_style else seq_mask_b
				plens_enc_att = plens + 1
				pseq_mask_b_enc_att = cat_zeros_at_start(pseq_mask_b)
				mpx_iter_enc_att_loss = attn_loss(result['dec_mpx']['attn_weights'], result['dec_mpx_iter']['attn_weights'], 'enc_attn_weights', 
					lens, plens_enc_att, seq_mask_b, pseq_mask_b_enc_att, self.att_loss_layers, self.iter_sym, self.att_l2_mode, size_average)
				loss_values['mpx_iter_enc_att'] = mpx_iter_enc_att_loss.item()
				loss_all = loss_all + scale * self.enc_att_scale * mpx_iter_enc_att_loss
		
		if 'dec_fm_iter' in result:
			scale = self.fm_iter_weight * up_weight
			fm_iter_loss = consistency_loss(result['dec_fm']['logits'], result['dec_fm_iter']['logits'], result['fm_mask_iter'], 
				self.iter_sym, self.l2_mode, size_average, self.cons_tau, self.cons_topk)
			loss_values['fm_iter'] = fm_iter_loss.item()
			loss_all = loss_all + scale * fm_iter_loss
			loss_values['fm_iter_acc'] = consistency_acc(result['dec_fm']['logits'], result['dec_fm_iter']['logits'], result['fm_mask_iter'], size_average)
			if self.self_att_scale > 0:
				fm_lens_self_att = result['fm_lens'] + 1
				fm_padding_mask_b_self_att = cat_zeros_at_start(result['fm_padding_mask_b'])
				fm_iter_self_att_loss = attn_loss(result['dec_fm']['attn_weights'], result['dec_fm_iter']['attn_weights'], 'self_attn_weights', 
					fm_lens_self_att, fm_lens_self_att, fm_padding_mask_b_self_att, fm_padding_mask_b_self_att, self.att_loss_layers, self.iter_sym, self.att_l2_mode, size_average)
				loss_values['fm_iter_self_att'] = fm_iter_self_att_loss.item()
				loss_all = loss_all + scale * self.self_att_scale * fm_iter_self_att_loss
			if self.enc_att_scale > 0:
				# lens_enc_att = lens+1 if self.enc_cat_style else lens
				# seq_mask_b_enc_att = cat_zeros_at_start(seq_mask_b) if self.enc_cat_style else seq_mask_b
				fm_lens_enc_att = result['fm_lens'] + 1
				fm_padding_mask_b_enc_att = cat_zeros_at_start(result['fm_padding_mask_b'])
				fm_iter_enc_att_loss = attn_loss(result['dec_fm']['attn_weights'], result['dec_fm_iter']['attn_weights'], 'enc_attn_weights', 
					lens, fm_lens_enc_att, seq_mask_b, fm_padding_mask_b_enc_att, self.att_loss_layers, self.iter_sym, self.att_l2_mode, size_average)
				loss_values['fm_iter_enc_att'] = fm_iter_enc_att_loss.item()
				loss_all = loss_all + scale * self.enc_att_scale * fm_iter_enc_att_loss
		
		if 'dec_fc' in result:
			scale = self.fc_tch_weight * up_weight
			if self.fc_tch_weight > 0 and arg_dict['up_stage']:
				fc_tch_loss = consistency_full_loss(result['dec_fc']['logits'], result['dec_fc']['fc_logits_target'], result['dec_fc']['fc_mask'], 
					False, self.l2_mode, size_average, self.cons_tau, self.cons_topk)
				loss_values['fc_tch'] = fc_tch_loss.item()
				loss_all = loss_all + scale * fc_tch_loss
			if 'fc_dctr' in result:
				fc_seq_mask = ~result['dec_fc']['fc_mask']
				fc_dctr_seq_masks = [~m for m in result['fc_dctr_masks']]
				if self.ctr_guide_scale > 0 and self.ctr_guide_delete and self.fc_ctr_guide:
					fc_dctr_guide_loss = 0
					for i in range(self.ctr_n):
						fc_dctr_guide_loss = fc_dctr_guide_loss + consistency_full_loss(result['fc_dctr'][i], result['dec_fc']['fc_logits_target'], result['fc_dctr_masks'][i], 
								False, self.l2_mode, size_average, self.cons_tau, self.cons_topk)
					loss_values['fc_dctr_guide'] = fc_dctr_guide_loss.item()
					loss_all = loss_all + scale * self.ctr_guide_scale * fc_dctr_guide_loss
				fc_dctr_loss = seq_contrast_loss(result['dec_fc']['logits'], result['fc_dctr'], 
									fc_seq_mask, fc_dctr_seq_masks, result['dec_fc']['fc_hard_target'], self.ctr_use_t1, self.ctr_use_t2, 
									self.ctr_fc_margin, self.ctr_by_seq, result['fc_dctr_word_ims'], result['fc_dctr_covs'], 
									result['fc_dctr_unchange'], size_average)
				loss_values['fc_dctr'] = fc_dctr_loss.item()
				loss_all = loss_all + self.ctr_weight * fc_dctr_loss
			if 'fc_ictr' in result:
				fc_seq_mask = ~result['dec_fc']['fc_mask']
				fc_ictr_seq_masks = [~m for m in result['fc_ictr_masks']]
				if self.ctr_guide_scale > 0 and self.ctr_guide_insert and self.fc_ctr_guide:
					fc_ictr_guide_loss = 0
					for i in range(self.ctr_n):
						fc_ictr_guide_loss = fc_ictr_guide_loss + consistency_full_loss(result['fc_ictr'][i], result['dec_fc']['fc_logits_target'], 
							result['dec_fc']['fc_mask'], False, self.l2_mode, size_average, self.cons_tau, self.cons_topk, 
							result['fc_ictr_filling_masks'][i])
					loss_values['fc_ictr_guide'] = fc_ictr_guide_loss.item()
					loss_all = loss_all + scale * self.ctr_guide_scale * fc_ictr_guide_loss
				fc_ictr_loss = seq_contrast_insert_loss(result['dec_fc']['logits'], result['fc_ictr'], 
									fc_seq_mask, fc_ictr_seq_masks, result['dec_fc']['fc_hard_target'], self.ctr_use_t1, 
									self.ctr_fc_margin, self.ctr_by_seq, result['fc_ictr_word_ims'], result['fc_ictr_covs'], 
									result['fc_ictr_filling_masks'], size_average)
				loss_values['fc_ictr'] = fc_ictr_loss.item()
				loss_all = loss_all + self.ctr_weight * fc_ictr_loss
		if 'bt' in result:
			scale = self.bt_weight * up_weight
			bt_loss = (masked_prediction_loss(result['bt']['logits'], x, mx_dec_mask, size_average, smooth=self.smooth) 
				if self.need_mx_dec 
				else seq_ce_logits_loss(result['bt']['logits'], x, para_lens, para_seq_mask, size_average, smooth=self.smooth))
			loss_values['bt'] = bt_loss.item()
			loss_all = loss_all + scale * bt_loss
			loss_values['bt_acc'] = (masked_prediction_acc(result['bt']['logits'], x, mx_dec_mask, size_average) 
				if self.need_mx_dec 
				else seq_acc(result['bt']['logits'], x, para_lens, para_seq_mask, size_average))

		if 'fc_bt' in result:
			scale = self.fc_bt_weight * up_weight
			fc_bt_loss = (masked_prediction_loss(result['fc_bt']['logits'], x, mx_dec_mask, size_average, smooth=self.smooth) 
				if self.need_mx_dec 
				else seq_ce_logits_loss(result['fc_bt']['logits'], x, para_lens, para_seq_mask, size_average, smooth=self.smooth))
			loss_values['fc_bt'] = fc_bt_loss.item()
			loss_all = loss_all + scale * fc_bt_loss
			loss_values['fc_bt_acc'] = (masked_prediction_acc(result['fc_bt']['logits'], x, mx_dec_mask, size_average) 
				if self.need_mx_dec 
				else seq_acc(result['fc_bt']['logits'], x, para_lens, para_seq_mask, size_average))
			if 'fc_bt_dctr' in result:
				if self.ctr_guide_scale > 0 and self.ctr_guide_delete:
					fc_bt_dctr_guide_loss = 0
					for i in range(self.ctr_n):
						fc_bt_dctr_guide_loss = fc_bt_dctr_guide_loss + (masked_prediction_loss(result['fc_bt_dctr'][i], x, result['fc_bt_dctr_masks'][i], size_average, smooth=self.smooth) 
										if self.need_mx_dec else 
										seq_ce_logits_loss(result['fc_bt_dctr'][i], x, 
										(~result['fc_bt_dctr_masks'][i]).long().sum(0), result['fc_bt_dctr_masks'][i], size_average, smooth=self.smooth))
					loss_values['fc_bt_dctr_guide'] = fc_bt_dctr_guide_loss.item()
					loss_all = loss_all + scale * self.ctr_guide_scale * fc_bt_dctr_guide_loss
				fc_bt_dctr_loss = (mx_contrast_loss(result['fc_bt']['logits'], result['fc_bt_dctr'], 
									mx_dec_mask, result['fc_bt_dctr_masks'], x, self.ctr_use_t1, self.ctr_use_t2, 
									self.ctr_margin, self.ctr_by_seq, result['fc_bt_dctr_word_ims'], [None]*self.ctr_n, 
									result['fc_bt_dctr_unchange'], size_average)
									if self.need_mx_dec else 
									seq_contrast_loss(result['fc_bt']['logits'], result['fc_bt_dctr'], 
									seq_mask, result['fc_bt_dctr_masks'], x, self.ctr_use_t1, self.ctr_use_t2, 
									self.ctr_margin, self.ctr_by_seq, result['fc_bt_dctr_word_ims'], [None]*self.ctr_n, 
									result['fc_bt_dctr_unchange'], size_average))
				loss_values['fc_bt_dctr'] = fc_bt_dctr_loss.item()
				loss_all = loss_all + self.ctr_weight * fc_bt_dctr_loss
			if 'fc_bt_ictr' in result:
				if self.ctr_guide_scale > 0 and self.ctr_guide_insert:
					fc_bt_ictr_guide_loss = 0
					for i in range(self.ctr_n):
						x_reform = reform_target(x, result['fc_bt_ictr_filling_masks'][i])
						fc_bt_ictr_guide_loss = fc_bt_ictr_guide_loss + (masked_prediction_loss_reform(result['fc_bt_ictr'][i], x_reform, 
							result['fc_bt_ictr_masks'][i], reform_mm(result['fc_bt_ictr_masks'][i], result['fc_bt_ictr_filling_masks'][i]), 
							size_average, smooth=self.smooth) 
							if self.need_mx_dec else seq_ce_logits_loss(result['fc_bt_ictr'][i], x_reform, 
							lens, reform_pm(result['fc_bt_ictr_masks'][i], result['fc_bt_ictr_filling_masks'][i]), 
							size_average, smooth=self.smooth))
					loss_values['fc_bt_ictr_guide'] = fc_bt_ictr_guide_loss.item()
					loss_all = loss_all + scale * self.ctr_guide_scale * fc_bt_ictr_guide_loss
				fc_bt_ictr_loss = (mx_contrast_insert_loss(result['fc_bt']['logits'], result['fc_bt_ictr'], 
									mx_dec_mask, result['fc_bt_ictr_masks'], x, self.ctr_use_t1, 
									self.ctr_margin, self.ctr_by_seq, result['fc_bt_ictr_word_ims'], [None]*self.ctr_n, 
									result['fc_bt_ictr_filling_masks'], size_average)
									if self.need_mx_dec else 
									seq_contrast_insert_loss(result['fc_bt']['logits'], result['fc_bt_ictr'], 
									seq_mask, result['fc_bt_ictr_masks'], x, self.ctr_use_t1, 
									self.ctr_margin, self.ctr_by_seq, result['fc_bt_ictr_word_ims'], [None]*self.ctr_n, 
									result['fc_bt_ictr_filling_masks'], size_average))
				loss_values['fc_bt_ictr'] = fc_bt_ictr_loss.item()
				loss_all = loss_all + self.ctr_weight * fc_bt_ictr_loss
		if self.ut_weight > 0:
			if ((self.fm_ut or self.fc_ut) and arg_dict['up_stage']) or (self.mpx_ut and not (self.mpx_ut_down and arg_dict['down_stage_over'])):
				discard_mask = get_discard_mask(x, px, lens, plens, seq_mask, self.ut_offset)
			if self.fm_ut and arg_dict['up_stage']:
				scale = self.ut_weight * up_weight
				fm_ut_loss = under_trans_loss(result['dec_fm']['logits'], ~result['fm_padding_mask'], x, discard_mask, size_average)
				loss_values['fm_ut'] = fm_ut_loss.item()
				loss_all = loss_all + scale * fm_ut_loss
			if self.fc_ut and arg_dict['up_stage']:
				scale = self.ut_weight * up_weight
				fc_ut_loss = under_trans_loss(result['dec_fc']['logits'], result['dec_fc']['fc_mask'], x, discard_mask, size_average)
				loss_values['fc_ut'] = fc_ut_loss.item()
				loss_all = loss_all + scale * fc_ut_loss
			if self.mpx_ut and not (self.mpx_ut_down and arg_dict['down_stage_over']):
				scale = (down_weight if self.mpx_ut_down else 1) * self.ut_weight
				mpx_ut_loss = under_trans_loss(result['dec_mpx']['logits'], mpx_dec_mask, x, discard_mask, size_average)
				loss_values['mpx_ut'] = mpx_ut_loss.item()
				loss_all = loss_all + scale * mpx_ut_loss

		
		if self.clf_weight > 0:
			if self.fm_clf and arg_dict['up_stage']:
				if self.fm_recomp_lens and self.less_len is None:
					loss_values['zl_rate'] = result['dec_fm']['hard_output_zl_mask'].float().mean().item() if size_average else result['dec_fm']['hard_output_zl_mask'].float().sum().item()
				scale = self.clf_weight * up_weight
				style_logits = self.style_train_tool(self.prepare_input_for_clf(self.style_train_tool, result['dec_fm'], True), soft_input = True)
				clf_loss = F.cross_entropy(style_logits, result['to_style'], reduction = 'mean' if size_average else 'sum')
				loss_values['clf'] = clf_loss.item()
				loss_all = loss_all + scale * clf_loss
				loss_values['style_acc'] = unit_acc(style_logits, result['to_style'], size_average)
			if self.mpx_clf and arg_dict['pre_up_stage'] and not (self.mpx_clf_down and arg_dict['down_stage_over']):
				scale = (down_weight if self.mpx_clf_down else 1) * self.clf_weight * pre_up_weight
				mpx_style_logits = self.style_train_tool(self.prepare_input_for_clf_from_mx(self.style_train_tool, result['dec_mpx'], batch_dict, True), soft_input = True)
				mpx_clf_loss = F.cross_entropy(mpx_style_logits, result['to_style'], reduction = 'mean' if size_average else 'sum')
				loss_values['mpx_clf'] = mpx_clf_loss.item()
				loss_all = loss_all + scale * mpx_clf_loss
				loss_values['mpx_style_acc'] = unit_acc(mpx_style_logits, result['to_style'], size_average)
			if self.fc_clf and arg_dict['up_stage']:
				scale = self.clf_weight * up_weight
				fc_style_logits = self.style_train_tool(self.prepare_input_for_clf_from_fc(self.style_train_tool, result['dec_fc'], True), soft_input = True)
				fc_clf_loss = F.cross_entropy(fc_style_logits, result['to_style'], reduction = 'mean' if size_average else 'sum')
				loss_values['fc_clf'] = fc_clf_loss.item()
				loss_all = loss_all + scale * fc_clf_loss
				loss_values['fc_style_acc'] = unit_acc(fc_style_logits, result['to_style'], size_average)
			
		if self.clf_weight > 0 and self.clf_adv:
			cal_fm_clf_adv = self.fm_clf_adv and arg_dict['up_stage']
			cal_mpx_clf_adv = self.mpx_clf_adv and arg_dict['pre_up_stage'] and not (self.mpx_clf_adv_down and arg_dict['down_stage_over'])
			cal_fc_clf_adv = self.fc_clf_adv and arg_dict['up_stage']
			scale = self.clf_weight
			src_logits = self.style_train_tool_update(self.prepare_input_for_clf_adv(self.style_train_tool, batch_dict))
			clf_adv_src_loss = F.cross_entropy(src_logits, style, reduction = 'mean' if size_average else 'sum')
			loss_values['clf_adv_src'] = clf_adv_src_loss.item()
			loss_all = loss_all + scale * self.clf_adv_scale * self.clf_adv_src_scale * clf_adv_src_loss
			loss_values['clf_adv_src_acc'] = unit_acc(src_logits, style, size_average)
			
			if cal_fm_clf_adv:
				scale = self.clf_weight * up_weight
				tsf_logits = self.style_train_tool_update(self.prepare_input_for_clf(self.style_train_tool, result['dec_fm'], False))
				clf_adv_tsf_loss = adv_loss(tsf_logits, style, result['to_style'], self.clf_adv_mode, size_average)
				loss_values['clf_adv_tsf'] = clf_adv_tsf_loss.item()
				loss_all = loss_all + scale * self.clf_adv_scale * clf_adv_tsf_loss
				loss_values['clf_adv_tsf_acc'] = unit_acc(tsf_logits, result['to_style'], size_average)
			if cal_mpx_clf_adv:
				scale = (down_weight if self.mpx_clf_adv_down else 1) * self.clf_weight * pre_up_weight
				mpx_tsf_logits = self.style_train_tool_update(self.prepare_input_for_clf_from_mx(self.style_train_tool, result['dec_mpx'], batch_dict, False))
				mpx_clf_adv_tsf_loss = adv_loss(mpx_tsf_logits, style, result['to_style'], self.clf_adv_mode, size_average)
				loss_values['mpx_clf_adv_tsf'] = mpx_clf_adv_tsf_loss.item()
				loss_all = loss_all + scale * self.clf_adv_scale * mpx_clf_adv_tsf_loss
				loss_values['mpx_clf_adv_tsf_acc'] = unit_acc(mpx_tsf_logits, result['to_style'], size_average)
			if cal_fc_clf_adv:
				scale = self.clf_weight * up_weight
				fc_tsf_logits = self.style_train_tool_update(self.prepare_input_for_clf_from_fc(self.style_train_tool, result['dec_fc'], False))
				fc_clf_adv_tsf_loss = adv_loss(fc_tsf_logits, style, result['to_style'], self.clf_adv_mode, size_average)
				loss_values['fc_clf_adv_tsf'] = fc_clf_adv_tsf_loss.item()
				loss_all = loss_all + scale * self.clf_adv_scale * fc_clf_adv_tsf_loss
				loss_values['fc_clf_adv_tsf_acc'] = unit_acc(fc_tsf_logits, result['to_style'], size_average)
		
		if self.lm_weight > 0:
			if self.fm_lm and arg_dict['up_stage']:
				scale = self.lm_weight * up_weight
				soft_outputs = result['dec_fm']['soft_outputs']
				lens_with_eos = result['dec_fm']['hard_outputs_lens_with_eos']
				padding_mask_with_eos = result['dec_fm']['hard_outputs_padding_mask_with_eos']
				start_tok = soft_outputs.new_zeros((1, soft_outputs.size(1), soft_outputs.size(2)))
				start_tok[:, :, constants.BOS_ID] = 1
				lm_logits = self.fluency_train_tool({'x': torch.cat([start_tok, soft_outputs[:-1]], 0), 'inds': result['to_style']}, True)
				lm_tgt = result['dec_fm']['hard_outputs'] if self.lm_tgt_mode == 'hard' else (soft_outputs if self.lm_tgt_mode == 'soft' else soft_outputs.detach())
				lm_loss = seq_ce_logits_loss(lm_logits, lm_tgt, lens_with_eos, padding_mask_with_eos, size_average)
				loss_values['lm'] = lm_loss.item()
				loss_all = loss_all + scale * lm_loss
				loss_values['ppl'] = math.exp(loss_values['lm'])
			if self.mpx_lm and arg_dict['pre_up_stage'] and not (self.mpx_lm_down and arg_dict['down_stage_over']):
				scale = (down_weight if self.mpx_lm_down else 1) * self.lm_weight * pre_up_weight
				soft_outputs = self.combine(result['dec_mpx'], px, mpx_dec_mask, True)
				lens_with_eos = plens
				padding_mask_with_eos = pseq_mask
				start_tok = soft_outputs.new_zeros((1, soft_outputs.size(1), soft_outputs.size(2)))
				start_tok[:, :, constants.BOS_ID] = 1
				lm_logits = self.fluency_train_tool({'x': torch.cat([start_tok, soft_outputs[:-1]], 0), 'inds': result['to_style']}, True)
				lm_tgt = self.combine(result['dec_mpx'], px, mpx_dec_mask, False) if self.lm_tgt_mode == 'hard' else (soft_outputs if self.lm_tgt_mode == 'soft' else soft_outputs.detach())
				mpx_lm_loss = seq_ce_logits_loss(lm_logits, lm_tgt, lens_with_eos, padding_mask_with_eos, size_average)
				loss_values['mpx_lm'] = mpx_lm_loss.item()
				loss_all = loss_all + scale * mpx_lm_loss
				loss_values['mpx_ppl'] = math.exp(loss_values['mpx_lm'])
		
		loss_values['loss_total'] = loss_all.item()
		return loss_all, loss_values
	
	def prepare_arg_dict(self):
		arg_dict = {}
		arg_dict['down_weight'] = rampdown(self.step, self.weight_down_start, self.weight_down_end, self.update_interval, self.down_alpha, True)
		arg_dict['up_weight'] = rampup(self.step, self.weight_up_start, self.weight_up_end, self.update_interval, self.up_alpha, True)
		arg_dict['pre_up_weight'] = rampup(self.step, self.pre_weight_up_start, self.pre_weight_up_end, self.update_interval, self.up_alpha, True)
		pre_up_stage = (self.pre_weight_up_start is None or self.step > self.pre_weight_up_start)
		up_stage = (self.weight_up_start is None or self.step > self.weight_up_start)
		down_stage_over = self.weight_down_start is not None and self.step > self.weight_down_end

		arg_dict['pre_up_stage'] = pre_up_stage
		arg_dict['up_stage'] = up_stage
		arg_dict['down_stage_over'] = down_stage_over

		arg_dict['extra_len'] = np.random.randint(-self.len_offset_l, self.len_offset_r+1) if self.dy_extra_len else self.extra_len

		arg_dict['rec'] = self.rec_weight > 0 and not (self.rec_down and down_stage_over)
		arg_dict['pseudo'] = self.pseudo_weight > 0 and not (self.pseudo_down and down_stage_over)
		arg_dict['bt'] = self.bt_weight > 0 and up_stage
		arg_dict['dec_use_px'] = self.simul_weight > 0 and not (self.simul_down and down_stage_over)

		arg_dict['fc_bt'] = self.fc_bt_weight > 0 and up_stage
		arg_dict['mpx_bt'] = self.mpx_bt_weight > 0 and pre_up_stage and not (self.mpx_bt_down and down_stage_over)

		arg_dict['dec_use_mpx_tch'] = self.tch_weight > 0 and pre_up_stage and not (self.tch_down and down_stage_over)
		arg_dict['dec_use_mpx_iter'] = self.mpx_iter_weight > 0 and pre_up_stage and not (self.mpx_iter_down and down_stage_over)
		arg_dict['mpx_out'] = (
			(self.clf_weight > 0 and self.mpx_clf and pre_up_stage and not (self.mpx_clf_down and down_stage_over))
			or (self.clf_weight > 0 and self.clf_adv and self.mpx_clf_adv and pre_up_stage and not (self.mpx_clf_adv_down and down_stage_over))
			or (self.lm_weight > 0 and self.mpx_lm and pre_up_stage and not (self.mpx_lm_down and down_stage_over))
			or arg_dict['dec_use_mpx_iter'] or arg_dict['mpx_bt'])
		arg_dict['dec_use_mpx'] = (
			(self.simul_mask_weight > 0 and not (self.simul_mask_down and down_stage_over)) or 
			(self.ut_weight > 0 and self.mpx_ut and not (self.mpx_ut_down and down_stage_over)) or 
			arg_dict['dec_use_mpx_tch'] or arg_dict['mpx_out'])
		
		arg_dict['dec_use_fm'] = (self.fm_iter_weight > 0 or self.bt_weight > 0 or 
			(self.clf_weight > 0 and self.fm_clf) 
			or (self.clf_weight > 0 and self.clf_adv and self.fm_clf_adv) 
			or (self.lm_weight > 0 and self.fm_lm)
			or (self.ut_weight > 0 and self.fm_ut)) and up_stage
		arg_dict['dec_use_fm_iter'] = self.fm_iter_weight > 0 and up_stage

		arg_dict['dec_use_fc'] = (self.fc_tch_weight > 0 or 
			(self.clf_weight > 0 and self.fc_clf) 
			or (self.clf_weight > 0 and self.clf_adv and self.fc_clf_adv)
			or (self.ut_weight > 0 and self.fc_ut)) and up_stage
		arg_dict['fc_out'] = (self.clf_weight > 0 and self.fc_clf and up_stage) or (
			self.clf_weight > 0 and self.clf_adv and self.fc_clf_adv and up_stage)

		arg_dict['rec_dctr'] = self.ctr_weight > 0 and self.step > self.ctr_start and arg_dict['rec'] and self.rec_ctr and self.ctr_delete
		arg_dict['px_dctr'] = self.ctr_weight > 0 and self.step > self.ctr_start and arg_dict['dec_use_px'] and self.px_ctr and self.ctr_delete
		arg_dict['mpx_dctr'] = self.ctr_weight > 0 and self.step > self.ctr_start and arg_dict['dec_use_mpx'] and self.mpx_ctr and self.ctr_delete
		arg_dict['mpx_bt_dctr'] = self.ctr_weight > 0 and self.step > self.ctr_start and arg_dict['mpx_bt'] and self.mpx_bt_ctr and self.ctr_delete
		arg_dict['fc_dctr'] = self.ctr_weight > 0 and self.step > self.ctr_start and arg_dict['dec_use_fc'] and self.fc_ctr and self.ctr_delete
		arg_dict['fc_bt_dctr'] = self.ctr_weight > 0 and self.step > self.ctr_start and arg_dict['fc_bt'] and self.fc_bt_ctr and self.ctr_delete

		arg_dict['rec_ictr'] = self.ctr_weight > 0 and self.step > self.ctr_start and arg_dict['rec'] and self.rec_ctr and self.ctr_insert
		arg_dict['px_ictr'] = self.ctr_weight > 0 and self.step > self.ctr_start and arg_dict['dec_use_px'] and self.px_ctr and self.ctr_insert
		arg_dict['mpx_ictr'] = self.ctr_weight > 0 and self.step > self.ctr_start and arg_dict['dec_use_mpx'] and self.mpx_ctr and self.ctr_insert
		arg_dict['mpx_bt_ictr'] = self.ctr_weight > 0 and self.step > self.ctr_start and arg_dict['mpx_bt'] and self.mpx_bt_ctr and self.ctr_insert
		arg_dict['fc_ictr'] = self.ctr_weight > 0 and self.step > self.ctr_start and arg_dict['dec_use_fc'] and self.fc_ctr and self.ctr_insert
		arg_dict['fc_bt_ictr'] = self.ctr_weight > 0 and self.step > self.ctr_start and arg_dict['fc_bt'] and self.fc_bt_ctr and self.ctr_insert
		
		arg_dict['tau'] = update_arg(self.tau, self.tau_up, self.step, self.update_interval, self.tau_update_start, self.tau_update_end, self.up_alpha, self.down_alpha)
		need_pseudo = (arg_dict['pseudo'] or arg_dict['dec_use_px'] or arg_dict['dec_use_mpx'])
		need_ptop = (arg_dict['dec_use_px'] or (self.simul_mask_weight > 0 and not (self.simul_mask_down and down_stage_over))
			) and self.px_hard_alpha < 1
		ignore_fields = []
		if not need_ptop:
			ignore_fields.extend(['topi', 'topp'])
		if not need_pseudo:
			ignore_fields.append('pseudo')
		arg_dict['ignore_fields'] = ignore_fields
		arg_dict['need_clf_adv'] = (self.clf_weight > 0 and self.clf_adv and self.fm_clf_adv and up_stage) or (
			self.clf_weight > 0 and self.clf_adv and self.mpx_clf_adv and pre_up_stage and not (self.mpx_clf_adv_down and down_stage_over))
		arg_dict.update(self.static_arg_dict)
		return arg_dict


	def train_batch(self, batch, arg_dict):
		b0 = debug_time_msg(self.debug)
		
		batch_dict = self.prepare_batch(batch)
		# self.optimizer.zero_grad()
		b1 = debug_time_msg(self.debug, b0, 'prepare batch')
		
		result = self.model.transfer(batch_dict, arg_dict)
		b2 = debug_time_msg(self.debug, b1, 'forward')
		
		loss, loss_values = self.compute_loss(result, batch_dict, arg_dict, True)
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
				# self.eval('valid', self.valid_loader)
				
				if (self.start_saving is None or self.step > self.start_saving):
					self.eval('test', self.test_loader, True)
					
				self.save_states('latest-')
				# to save space
				if self.step != self.eval_interval:
					old_model_path = os.path.join(self.model_path, 'latest-model-{}'.format(self.step - self.eval_interval))
					# os.remove(os.path.join(self.model_path, 'latest-model-{}'.format(self.step - self.eval_interval)))
					if os.path.isfile(old_model_path):
						os.remove(old_model_path)

		self.model.train()
		data_iters = [iter(tl) for tl in self.train_loaders]
		accl = OrderedDict()
		never_set_flag = True
		while self.step <= self.n_iters:
			arg_dict = self.prepare_arg_dict()
			if never_set_flag and not arg_dict['rec']:
				never_set_flag = False
				self.train_loaders[0].dataset.fields['text'].postprocessing.turn_off_noise()

			
			if self.batch_merge:
				prepare_optimize(arg_dict['need_clf_adv'], accl)
				update_flag = True
				b0 = debug_time_msg(self.debug)
				batch = [next(data_iter) for data_iter in data_iters]
				batch = list(chain.from_iterable(batch))
				batch = Batch(batch, self.train_loaders[0].dataset, self.device, arg_dict['ignore_fields'])
				b1 = debug_time_msg(self.debug, b0, 'read data')
				try:
					loss_values = self.train_batch(batch, arg_dict)
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
					optimize(arg_dict['need_clf_adv'], accl, 1)
					self.step += 1

			else:
				if self.opt_accross_styles:
					prepare_optimize(arg_dict['need_clf_adv'], accl)
				update_flag = True
				i = 0
				while i < len(data_iters):
					if not self.opt_accross_styles:
						prepare_optimize(arg_dict['need_clf_adv'], accl)
					b0 = debug_time_msg(self.debug)
					
					batch = next(data_iters[i])
					batch = Batch(batch, self.train_loaders[i].dataset, self.device, arg_dict['ignore_fields'])
					b1 = debug_time_msg(self.debug, b0, 'read data')
					try:
						loss_values = self.train_batch(batch, arg_dict)
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
						optimize(arg_dict['need_clf_adv'], accl, 1)
						self.step += 1
					i += 1
				if self.opt_accross_styles and update_flag:
					optimize(arg_dict['need_clf_adv'], accl, len(data_iters))
					self.step += 1
		self.summary_writer.close()

	def fluency_eval(self, result, to_style, name='hard_outputs', 
		padding_name='hard_outputs_padding_mask_with_eos', len_name='hard_outputs_lens_with_eos'):
		hard_outputs = result[name]
		hard_outputs_lens_with_eos = result[len_name]
		hard_outputs_padding_mask_with_eos = result[padding_name]
		bsz = hard_outputs.size(1)
		start_tok = hard_outputs.new_full((1, bsz), constants.BOS_ID)
		fluency_eval_nll = seq_ce_logits_loss(
			self.fluency_eval_tool({'x': torch.cat([start_tok, hard_outputs[:-1]], 0), 'inds': to_style}), 
			hard_outputs, hard_outputs_lens_with_eos, hard_outputs_padding_mask_with_eos, False).item()
		if self.bi_fluency_eval:
			hard_outputs_rev = reverse_seq(hard_outputs, hard_outputs_lens_with_eos-1, hard_outputs==constants.EOS_ID)
			fluency_eval_nll = fluency_eval_nll + seq_ce_logits_loss(
				self.fluency_eval_tool_rev({'x': torch.cat([start_tok, hard_outputs_rev[:-1]], 0), 'inds': to_style}), 
				hard_outputs_rev, hard_outputs_lens_with_eos, hard_outputs_padding_mask_with_eos, False).item()
			fluency_eval_nll = fluency_eval_nll / 2
		return fluency_eval_nll

	def eval(self, name, dataset_loader, with_truth = False):
		if not self.eval_dropout:
			self.model.eval()
		if self.clf_weight > 0 and not self.aux_model_eval_mode:
			self.style_train_tool.eval()
		n_total = len(dataset_loader.dataset)
		accl = OrderedDict()
		start = time.time()
		trans_sens = []
		if self.cal_inter:
			if self.all_iter_eval:
				beam_size = self.len_offset_l + self.len_offset_r + 1
				accl_inter = [[OrderedDict() for j in range(self.iter_num)] for k in range(beam_size)]
				trans_sens_inter = [[[] for j in range(self.iter_num)] for k in range(beam_size)]
				if self.save_details:
					masked_sens = [[[] for j in range(self.iter_num-1)] for k in range(beam_size)]
					scores = [[[] for j in range(self.iter_num)] for k in range(beam_size)]
					final_scores = [[[] for j in range(self.iter_num)] for k in range(beam_size)]
					best_offset = []
					best_iter = []
			else:
				accl_inter = [OrderedDict() for j in range(self.iter_num-1)]
				trans_sens_inter = [[] for j in range(self.iter_num-1)]
				if self.save_details:
					masked_sens = [[] for j in range(self.iter_num-1)]
					scores = [[] for j in range(self.iter_num)]

		self.train_loaders[0].dataset.fields['text'].postprocessing.to_eval_mode()
		ignore_fields = ['pseudo', 'topi', 'topp']
		
		with torch.no_grad():
			for i, batch in enumerate(dataset_loader):
				batch_dict = self.prepare_batch(Batch(batch, dataset_loader.dataset, self.device, ignore_fields))
				result = self.model.iterative_transfer(batch_dict, self.fm_fill_eos,
					self.len_offset_l, self.len_offset_r, self.dec_mask_mode, self.mask_repeat,
					self.simple_lp, self.lp_value, self.lp_rela, 
					self.lp_cb_rela, self.lp_cb_add, self.lp_cb_simple, self.lp_cb_value, 
					self.iter_num, self.emit_mi, self.ret_mi, self.mi_alpha, self.all_iter_eval, self.rescore_mode,
					self.style_train_tool if self.clf_weight > 0 else None, self.rescore_beta, self.rescore_at_model if self.rescore_mode == 'at' else None,
					self.add_cov, self.cov_mode, self.cov_weight, self.cov_inv, self.cov_with_start)
				
				style_eval_acc = unit_acc(self.style_eval_tool(self.prepare_input_for_clf(self.style_eval_tool, result, False)), 
					result['to_style'], False)
				fluency_eval_nll = self.fluency_eval(result, result['to_style'])
				update_accl(accl, {'style_eval_acc': style_eval_acc, 'fluency_eval_nll': fluency_eval_nll})
				to_sentences(result['hard_outputs'], result['hard_outputs_lens_with_eos']-1, self.itos, trans_sens)
				
				if self.cal_inter:
					if self.all_iter_eval:
						for k in range(beam_size):
							for j in range(self.iter_num):
								key_suffix = f'offset_{k - self.len_offset_l}_iter_{j}'
								style_eval_acc = unit_acc(self.style_eval_tool(
									self.prepare_input_for_clf(self.style_eval_tool, result, False, f'outputs_{key_suffix}',
									f'padding_{key_suffix}', f'padding_b_{key_suffix}', f'lens_{key_suffix}')), 
									result['to_style'], False)
								fluency_eval_nll = self.fluency_eval(result, result['to_style'], f'outputs_{key_suffix}',
									f'padding_{key_suffix}', f'lens_{key_suffix}')
								update_accl(accl_inter[k][j], {'style_eval_acc': style_eval_acc, 'fluency_eval_nll': fluency_eval_nll})
								to_sentences(result[f'outputs_{key_suffix}'], result[f'lens_{key_suffix}']-1, self.itos, trans_sens_inter[k][j])
								if self.save_details:
									if j < self.iter_num - 1:
										to_arrs(result[f'masked_outputs_{key_suffix}'], result[f'lens_{key_suffix}']-1, masked_sens[k][j])
									to_arrs(result[f'scores_{key_suffix}'], result[f'lens_{key_suffix}']-1, scores[k][j])
									to_values(result[f'final_scores_{key_suffix}'], final_scores[k][j])
						if self.save_details:
							to_values(result['best_offset'], best_offset)
							to_values(result['best_iter'], best_iter)
					else:
						for j in range(self.iter_num-1):
							style_eval_acc = unit_acc(self.style_eval_tool(
								self.prepare_input_for_clf(self.style_eval_tool, result, False, f'outputs_iter_{j}')), 
								result['to_style'], False)
							fluency_eval_nll = self.fluency_eval(result, result['to_style'], f'outputs_iter_{j}')
							update_accl(accl_inter[j], {'style_eval_acc': style_eval_acc, 'fluency_eval_nll': fluency_eval_nll})
							to_sentences(result[f'outputs_iter_{j}'], result['hard_outputs_lens_with_eos']-1, self.itos, trans_sens_inter[j])
							if self.save_details:
								to_arrs(result[f'masked_outputs_iter_{j}'], result['hard_outputs_lens_with_eos']-1, masked_sens[j])
								to_arrs(result[f'scores_iter_{j}'], result['hard_outputs_lens_with_eos']-1, scores[j])
						if self.save_details:
							to_arrs(result[f'scores_iter_{self.iter_num-1}'], result['hard_outputs_lens_with_eos']-1, scores[self.iter_num-1])

					
					
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
		if self.cal_inter:
			if self.all_iter_eval:
				for k in range(beam_size):
					for j in range(self.iter_num):
						key_suffix = f'-offset-{k - self.len_offset_l}-iter-{j}'
						get_final_accl(accl_inter[k][j], n_total)
						accl_inter[k][j]['fluency_eval_nppl'] = - math.exp(accl_inter[k][j]['fluency_eval_nll'])
						trans_sens_inter[k][j] = reorder(dataset_loader.order, trans_sens_inter[k][j])
						accl_inter[k][j]['self_bleu'], accl_inter[k][j]['human_bleu'] = self.save_outputs_and_compute_bleu(name, trans_sens_inter[k][j], with_truth, key_suffix)
						if not self.eval_only:
							add_to_writer(accl_inter[k][j], self.step, f'{name}{key_suffix}', self.summary_writer)
						print('{} intermediate performance at step {} / offset {} / iteration {}'.format(name, self.step, k - self.len_offset_l, j))
						print_loss(accl_inter[k][j])
				if self.save_details:
					for k in range(beam_size):
						for j in range(self.iter_num):
							scores[k][j] = reorder(dataset_loader.order, scores[k][j])
							final_scores[k][j] = reorder(dataset_loader.order, final_scores[k][j])
							if j < self.iter_num - 1:
								masked_sens[k][j] = reorder(dataset_loader.order, masked_sens[k][j])
					best_offset = reorder(dataset_loader.order, best_offset)
					best_iter = reorder(dataset_loader.order, best_iter)
					detail_file = os.path.join(self.output_path, '{}-result-{}.all-detail'.format(name, self.step))
					save_all_iter_details(src_sens, trans_sens_inter, masked_sens, scores, final_scores, best_offset, best_iter, self.len_offset_l, detail_file)
			else:
				for j in range(self.iter_num-1):
					get_final_accl(accl_inter[j], n_total)
					accl_inter[j]['fluency_eval_nppl'] = - math.exp(accl_inter[j]['fluency_eval_nll'])
					trans_sens_inter[j] = reorder(dataset_loader.order, trans_sens_inter[j])
					accl_inter[j]['self_bleu'], accl_inter[j]['human_bleu'] = self.save_outputs_and_compute_bleu(name, trans_sens_inter[j], with_truth, f'-iter-{j}')
					if not self.eval_only:
						add_to_writer(accl_inter[j], self.step, f'{name}-iter-{j}', self.summary_writer)
					print('{} intermediate performance at step {} / iteration {}'.format(name, self.step, j))
					print_loss(accl_inter[j])
				if self.save_details:
					for j in range(self.iter_num-1):
						masked_sens[j] = reorder(dataset_loader.order, masked_sens[j])
					for j in range(self.iter_num):
						scores[j] = reorder(dataset_loader.order, scores[j])
					detail_file = os.path.join(self.output_path, '{}-result-{}.detail'.format(name, self.step))
					save_iter_details(src_sens, trans_sens, trans_sens_inter, masked_sens, scores, detail_file)


		print('{:.2f} s for evaluation'.format(time.time() - start))
		if not self.eval_dropout:
			self.model.train()
		if self.clf_weight > 0 and not self.aux_model_eval_mode:
			self.style_train_tool.train()
		self.train_loaders[0].dataset.fields['text'].postprocessing.to_train_mode()
		

	def save_outputs_and_compute_bleu(self, name, trans_sens, with_truth, suffix=''):
		stats = self.dataset_stats[name]
		cur_ind = 0
		self_bleu = 0
		if with_truth:
			human_bleu = 0
		
		for i in range(self.num_classes):
			output_file = os.path.join(self.output_path, '{}-result-{}{}.{}'.format(name, self.step, suffix, i))
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

	
		
if __name__=='__main__':

	parser=argparse.ArgumentParser()
	parser.add_argument('version', type=str)
	parser.add_argument('-seed', type=int, default=1)
	parser.add_argument('-gpu', type=int, default=0)
	parser.add_argument('-debug', type=str2bool, default=False)
	parser.add_argument('-cudnn_enabled', type=str2bool, default=True)
	parser.add_argument('-eval_only', type=str2bool, default=False)
	parser.add_argument('-eval_dropout', type=str2bool, default=False)
	parser.add_argument('-cal_inter', type=str2bool, default=False, nargs='?')
	parser.add_argument('-save_details', type=str2bool, default=False, nargs='?')
	parser.add_argument('-keep_only_full', type=str2bool, default=True)
	parser.add_argument('-label_split', type=str2bool, default=True)
	parser.add_argument('-load_optim', type=str2bool, default=True)
	parser.add_argument('-batch_merge', type=str2bool, default=False)
	parser.add_argument('-opt_accross_styles', type=str2bool, default=False, nargs='?')
	parser.add_argument('-train_from', type=str, default=None, nargs='?')
	parser.add_argument('-train_from_at', type=str, default=None, nargs='?')
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
	parser.add_argument('-dec_mask_mode', type=str2bool, default=False)
	parser.add_argument('-para_x_fm', type=str2bool, default=False, nargs='?')
	parser.add_argument('-mask_tgt_consc', type=str2bool, default=False, nargs='?')
	parser.add_argument('-mask_tgt_span', type=str2bool, default=False, nargs='?')
	parser.add_argument('-mask_tgt_span_len', type=int, default=2, nargs='?')
	
	parser.add_argument('-emb_size', type=int, default=200)
	parser.add_argument('-emb_max_norm', type=float, default=1.0, nargs='?')
	parser.add_argument('-hid_size', type=int, default=200)
	parser.add_argument('-num_heads', type=int, default=8)
	parser.add_argument('-enc_num_layers', type=int, default=1)
	parser.add_argument('-dec_num_layers', type=int, default=1)
	parser.add_argument('-diff_bias', type=str2bool, default=True)
	parser.add_argument('-pos_sincode', type=str2bool, default=True)
	parser.add_argument('-token_emb_scale', type=str2bool, default=True)
	parser.add_argument('-dropout_rate', type=float, default=0.0)
	parser.add_argument('-att_dropout_rate', type=float, default=0)
	parser.add_argument('-transformer_norm_bf', type=str2bool, default=False)
	parser.add_argument('-enc_cat_style', type=str2bool, default=False)
	parser.add_argument('-share_pos_emb', type=str2bool, default=True)
	parser.add_argument('-positional_att', type=str2bool, default=True)
	parser.add_argument('-apply_self_mask', type=str2bool, default=True)
	parser.add_argument('-self_mask_escape_start', type=str2bool, default=True, nargs='?')
	parser.add_argument('-fm_fill_eos', type=str2bool, default=True, nargs='?')
	parser.add_argument('-tch_keep_rate', type=float, default=0.2, nargs='?')
	parser.add_argument('-iter_keep_rate', type=float, default=0.2, nargs='?')
	parser.add_argument('-iter_random', type=str2bool, default=True, nargs='?')
	parser.add_argument('-fm_use_pseudo_len', type=str2bool, default=False, nargs='?')
	parser.add_argument('-fm_recomp_lens', type=str2bool, default=False, nargs='?')
	parser.add_argument('-less_len', type=int, default=None, nargs='?')
	parser.add_argument('-bt_use_recomp_lens', type=str2bool, default=False, nargs='?')
	parser.add_argument('-dy_extra_len', type=str2bool, default=True)
	parser.add_argument('-extra_len', type=int, default=0, nargs='?')
	parser.add_argument('-greedy_train', type=str2bool, default=True)

	parser.add_argument('-len_offset_l', type=int, default=2)
	parser.add_argument('-len_offset_r', type=int, default=2)
	parser.add_argument('-simple_lp', type=str2bool, default=True)
	parser.add_argument('-lp_value', type=float, default=1.0)
	parser.add_argument('-lp_rela', type=str2bool, default=False)
	parser.add_argument('-lp_cb_rela', type=str2bool, default=False)
	parser.add_argument('-lp_cb_add', type=str2bool, default=False, nargs='?')
	parser.add_argument('-lp_cb_simple', type=str2bool, default=False, nargs='?')
	parser.add_argument('-lp_cb_value', type=float, default=1.0, nargs='?')
	parser.add_argument('-mask_repeat', type=str2bool, default=False, nargs='?')
	parser.add_argument('-iter_num', type=int, default=4, nargs='?')
	parser.add_argument('-emit_mi', type=str2bool, default=False)
	parser.add_argument('-ret_mi', type=str2bool, default=False)
	parser.add_argument('-mi_alpha', type=float, default=1.0, nargs='?')
	parser.add_argument('-all_iter_eval', type=str2bool, default=False)
	parser.add_argument('-rescore_mode', type=str, default=None, nargs='?')
	parser.add_argument('-rescore_beta', type=float, default=1.0, nargs='?')
	parser.add_argument('-rescore_at_model', type=str, default=None, nargs='?')
	parser.add_argument('-add_cov', type=str2bool, default=False)
	parser.add_argument('-cov_mode', type=str, default=None, nargs='?')
	parser.add_argument('-cov_weight', type=float, default=1.0, nargs='?')
	parser.add_argument('-cov_inv', type=str2bool, default=True, nargs='?')
	parser.add_argument('-cov_with_start', type=str2bool, default=False, nargs='?')

	parser.add_argument('-ctr_weight', type=float, default=0.0)
	parser.add_argument('-ctr_start', type=int, default=None, nargs='?')
	parser.add_argument('-ctr_guide_scale', type=float, default=0.0, nargs='?')
	parser.add_argument('-ctr_delete', type=str2bool, default=True, nargs='?')
	parser.add_argument('-ctr_insert', type=str2bool, default=False, nargs='?')
	parser.add_argument('-ctr_guide_delete', type=str2bool, default=True, nargs='?')
	parser.add_argument('-ctr_guide_insert', type=str2bool, default=False, nargs='?')
	parser.add_argument('-rec_ctr', type=str2bool, default=False, nargs='?')
	parser.add_argument('-px_ctr', type=str2bool, default=False, nargs='?')
	parser.add_argument('-mpx_ctr', type=str2bool, default=False, nargs='?')
	parser.add_argument('-mpx_bt_ctr', type=str2bool, default=False, nargs='?')
	parser.add_argument('-fc_ctr', type=str2bool, default=False, nargs='?')
	parser.add_argument('-fc_ctr_guide', type=str2bool, default=True, nargs='?')
	parser.add_argument('-fc_bt_ctr', type=str2bool, default=False, nargs='?')
	parser.add_argument('-ctr_n', type=int, default=1, nargs='?')
	parser.add_argument('-ctr_kmin', type=int, default=1, nargs='?')
	parser.add_argument('-ctr_kmax', type=int, default=1, nargs='?')
	parser.add_argument('-ctr_fc_good_cov', type=str2bool, default=False, nargs='?')
	parser.add_argument('-ctr_fc_bad_cov', type=str2bool, default=False, nargs='?')
	parser.add_argument('-ctr_margin', type=float, default=1, nargs='?')
	parser.add_argument('-ctr_fc_margin', type=float, default=1, nargs='?')
	parser.add_argument('-ctr_word_im', type=str2bool, default=False, nargs='?')
	parser.add_argument('-ctr_by_seq', type=str2bool, default=False, nargs='?')
	parser.add_argument('-ctr_use_t1', type=str2bool, default=False, nargs='?')
	parser.add_argument('-ctr_use_t2', type=str2bool, default=False, nargs='?')



	parser.add_argument('-tau', type=float, default=0.5)
	parser.add_argument('-tau_up', type=str2bool, default=None, nargs='?')
	parser.add_argument('-tau_update_start', type=int, default=None, nargs='?')
	parser.add_argument('-tau_update_end', type=int, default=None, nargs='?')
	
	parser.add_argument('-rec_weight', type=float, default=1.0)
	parser.add_argument('-pseudo_weight', type=float, default=0.0)
	parser.add_argument('-bt_weight', type=float, default=1.0)
	parser.add_argument('-clf_weight', type=float, default=1.0)
	parser.add_argument('-lm_weight', type=float, default=0.0)
	parser.add_argument('-enc_pred_weight', type=float, default=0.0)
	parser.add_argument('-lm_tgt_mode', type=str, default='hard', nargs='?', choices=['soft', 'soft_detach', 'hard'])
	parser.add_argument('-simul_weight', type=float, default=0.0)
	parser.add_argument('-simul_mask_weight', type=float, default=1.0)
	parser.add_argument('-tch_weight', type=float, default=1.0)
	parser.add_argument('-mpx_iter_weight', type=float, default=1.0)
	parser.add_argument('-fm_iter_weight', type=float, default=1.0)
	parser.add_argument('-clf_adv', type=str2bool, default=False, nargs='?')
	parser.add_argument('-clf_adv_mode', type=str, default='ac', nargs='?')
	parser.add_argument('-clf_adv_scale', type=float, default=1.0, nargs='?')
	parser.add_argument('-clf_adv_lr', type=float, default=0.001, nargs='?')
	parser.add_argument('-clf_adv_src_scale', type=float, default=1.0, nargs='?')
	parser.add_argument('-fm_clf', type=str2bool, default=True, nargs='?')
	parser.add_argument('-mpx_clf', type=str2bool, default=False, nargs='?')
	parser.add_argument('-fm_clf_adv', type=str2bool, default=True, nargs='?')
	parser.add_argument('-mpx_clf_adv', type=str2bool, default=False, nargs='?')

	parser.add_argument('-fc_tch_weight', type=float, default=0.0)
	parser.add_argument('-fc_bt_weight', type=float, default=0.0)
	parser.add_argument('-fc_clf', type=str2bool, default=False, nargs='?')
	parser.add_argument('-fc_clf_adv', type=str2bool, default=False, nargs='?')
	parser.add_argument('-mpx_bt_weight', type=float, default=0.0)
	parser.add_argument('-mpx_bt_down', type=str2bool, default=False, nargs='?')
	parser.add_argument('-fc_iter_num', type=int, default=4, nargs='?')
	parser.add_argument('-fc_mask_rate', type=float, default=0.5, nargs='?')
	parser.add_argument('-fc_mask_mode', type=str, default='random', nargs='?', choices=['random', 'topk'])
	parser.add_argument('-fc_mask_largest', type=str2bool, default=False, nargs='?')
	parser.add_argument('-cons_tau', type=float, default=1.0, nargs='?')
	parser.add_argument('-cons_topk', type=int, default=None, nargs='?')

	parser.add_argument('-ut_weight', type=float, default=0.0)
	parser.add_argument('-fm_ut', type=str2bool, default=True, nargs='?')
	parser.add_argument('-mpx_ut', type=str2bool, default=False, nargs='?')
	parser.add_argument('-fc_ut', type=str2bool, default=False, nargs='?')
	parser.add_argument('-mpx_ut_down', type=str2bool, default=False, nargs='?')
	parser.add_argument('-ut_offset', type=int, default=3, nargs='?')

	parser.add_argument('-fm_lm', type=str2bool, default=True, nargs='?')
	parser.add_argument('-mpx_lm', type=str2bool, default=False, nargs='?')
	parser.add_argument('-px_hard_alpha', type=float, default=1.0, nargs='?')
	parser.add_argument('-self_att_scale', type=float, default=0.0, nargs='?')
	parser.add_argument('-enc_att_scale', type=float, default=0.0, nargs='?')
	parser.add_argument('-bt_sg', type=str2bool, default=True, nargs='?')
	parser.add_argument('-tch_sym', type=str2bool, default=False, nargs='?')
	parser.add_argument('-iter_sym', type=str2bool, default=False, nargs='?')
	parser.add_argument('-l2_mode', type=str2bool, default=True, nargs='?')
	parser.add_argument('-att_l2_mode', type=str2bool, default=True, nargs='?')
	parser.add_argument('-att_loss_layers', type=int, default=1, nargs='?')
	parser.add_argument('-smooth', type=float, default=0.0)

	parser.add_argument('-start_saving', type=int, default=None, nargs='?')
	parser.add_argument('-weight_up_start', type=int, default=None, nargs='?')
	parser.add_argument('-weight_up_end', type=int, default=None, nargs='?')
	parser.add_argument('-pre_weight_up_start', type=int, default=None, nargs='?')
	parser.add_argument('-pre_weight_up_end', type=int, default=None, nargs='?')
	parser.add_argument('-weight_down_start', type=int, default=None, nargs='?')
	parser.add_argument('-weight_down_end', type=int, default=None, nargs='?')
	parser.add_argument('-rec_down', type=str2bool, default=False, nargs='?')
	parser.add_argument('-pseudo_down', type=str2bool, default=False, nargs='?')
	parser.add_argument('-simul_down', type=str2bool, default=False, nargs='?')
	parser.add_argument('-simul_mask_down', type=str2bool, default=False, nargs='?')
	parser.add_argument('-tch_down', type=str2bool, default=False, nargs='?')
	parser.add_argument('-mpx_iter_down', type=str2bool, default=False, nargs='?')
	parser.add_argument('-mpx_clf_down', type=str2bool, default=False, nargs='?')
	parser.add_argument('-mpx_clf_adv_down', type=str2bool, default=False, nargs='?')
	parser.add_argument('-mpx_lm_down', type=str2bool, default=False, nargs='?')
	
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
	parser.add_argument('-lr_warmup_steps', type=int, default=4000)
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
	if config.eval_only:
		trainer.eval('test', trainer.test_loader, True)
	else:
		trainer.train()
	print('Finish time: ', time.strftime('%X %x %Z'))
