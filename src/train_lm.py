import time
import sys
import os
from collections import namedtuple
import math
import random
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from datalib.iterator import BucketIterator
from datalib.batch import Batch
from data_loader import Corpus
from train_utils import *
from loss import *
from layers import get_padding_mask
from lms import rnn_lm, Transformer_lm
from torch.utils.tensorboard import SummaryWriter



Record = namedtuple('Record', 'step ppl the_other', defaults = (0, float('inf'), float('inf')))

class Solver(object):
	"""docstring for Solver"""
	def __init__(self, config):
		super(Solver, self).__init__()
		self.select_label = config.select_label
		self.max_grad_norm = config.max_grad_norm
		# self.eps = config.eps
		# self.lr = config.lr
		# self.update_interval = config.update_interval
		# self.up_alpha = config.up_alpha
		# self.down_alpha = config.down_alpha
		# self.lr_up_start = config.lr_up_start
		# self.lr_up_end = config.lr_up_end
		# self.lr_down_start = config.lr_down_start
		# self.lr_down_end = config.lr_down_end
		self.n_iters = config.n_iters
		self.log_interval = config.log_interval
		self.eval_interval = config.eval_interval
		self.eval_only = config.eval_only
		self.device = torch.device('cuda', config.gpu)

		datasets = Corpus.iters_dataset_simple(config.work_dir, config.dataset, 'lm', config.max_sen_len,
							config.min_freq, config.max_vocab_size, reverse=config.reverse)
		if config.ignore_splits:
			# for classifier training, we don't use the test split for style transfer as it is too small
			# so we split the training set to new dev and test
			datasets = datasets[0].split([0.8, 0.1, 0.1], True)
		if self.select_label is not None:
			datasets = zip(*[d.stratify_split('label') if d is not None else None for d in datasets])
			datasets = list(datasets)[self.select_label]
		self.train_loader, self.valid_loader, self.test_loader = BucketIterator.splits(datasets, 
							batch_sizes = [config.batch_size, config.eval_batch_size, config.eval_batch_size], 
							retain_order = False)

		dir_name = 'lm_'+('all' if self.select_label is None else str(self.select_label)) + ('_r' if config.reverse else '')

		self.model_path = os.path.join(config.work_dir, 'model', config.dataset, 'lm', 
										config.version)
		self.summary_path = os.path.join(config.work_dir, 'summary', config.dataset, 'lm', 
										config.version)
		makedirs(self.model_path)
		makedirs(self.summary_path)
		if not (self.eval_only or config.save_only):
			self.summary_writer = SummaryWriter(self.summary_path)

		vocab_size = len(self.train_loader.dataset.fields['text'].vocab)

		print('train size:', len(self.train_loader.dataset))
		print('valid size:', len(self.valid_loader.dataset))
		print('test size:', len(self.test_loader.dataset))
		print('vocab size:', vocab_size)

		num_styles = len(self.train_loader.dataset.fields['label'].vocab)
		print('number of styles:', num_styles)
		self.diff_bias = self.select_label is None and config.diff_bias == True

		self.model_type = config.model_type

		if config.model_type == 'rnn':
			self.args = [vocab_size, config.emb_size, config.emb_max_norm, config.hid_size, config.rnn_type, config.dropout_rate, 
				num_styles if self.diff_bias else 1, config.tie_weights]
			self.model = rnn_lm(vocab_size, config.emb_size, config.emb_max_norm, config.hid_size, config.rnn_type, config.dropout_rate, 
				num_styles if self.diff_bias else 1, config.tie_weights)
		else:
			self.args = [vocab_size, config.emb_size, config.emb_max_norm, config.pos_sincode, config.token_emb_scale, config.max_sen_len,
				config.num_heads, config.hid_size, config.num_layers,
				config.dropout_rate, num_styles if self.diff_bias else 1, config.tie_weights, config.att_dropout_rate, config.transformer_norm_bf]
			self.model = Transformer_lm(vocab_size, config.emb_size, config.emb_max_norm, config.pos_sincode, config.token_emb_scale, config.max_sen_len,
				config.num_heads, config.hid_size, config.num_layers,
				config.dropout_rate, num_styles if self.diff_bias else 1, config.tie_weights, config.att_dropout_rate, config.transformer_norm_bf)
		if config.pretrained_emb is not None:
			text_vocab = self.train_loader.dataset.fields['text'].vocab
			text_vocab.load_vectors(config.pretrained_emb, cache=os.path.join(config.work_dir, 'word_vectors'), max_vectors=config.pretrained_emb_max)
			self.model.emb.weight.data.copy_(text_vocab.vectors)
			text_vocab.vectors = None
		self.model.to(self.device)
		self.optimizer = build_optimizer(config.optim_method, self.model, config.lr, config.momentum, config.weight_decay, config.beta2)
		self.lr_scheduler = build_lr_scheduler(config.lr_method, self.optimizer, config.lr_warmup_steps, 
												config.lr_decay_steps, config.lr_decay_mode, config.lr_min_factor, config.lr_decay_rate, config.lr_init_factor)
		self.step = 1
		self.best_results = {'valid': Record(), 'test': Record()}
		if config.train_from is not None:
			check_point=torch.load(config.train_from, map_location=lambda storage, loc: storage)
			self.model.load_state_dict(check_point['model_state'])
			self.optimizer.load_state_dict(check_point['optimizer_state'])
			self.lr_scheduler.load_state_dict(check_point['lr_scheduler_state'])
			self.step = check_point['step']
			self.best_results = check_point['best_results']
			del check_point

		

	def save_states(self, prefix = ''):
		check_point = {
			'args': self.args,
			'model_type': self.model_type,
			'step': self.step,
			'model_state': self.model.state_dict(),
			'optimizer_state': self.optimizer.state_dict(),
			'lr_scheduler_state': self.lr_scheduler.state_dict(),
			'best_results': self.best_results
		}
		filename = os.path.join(self.model_path, '{}model-{}'.format(prefix, self.step))
		torch.save(check_point, filename)

	def save_model(self):
		check_point = {
			'model': self.model
		}
		filename = os.path.join(self.model_path, 'full-model-{}'.format(self.step))
		torch.save(check_point, filename)
		
	def prepare_batch(self, batch):
		full_text, lens = batch.text
		x, t = full_text[:-1], full_text[1:]
		lens.sub_(1)
		style = batch.label if self.diff_bias else None
		padding_mask = get_padding_mask(x, lens)
		return {'x': x, 'lens': lens, 'enc_padding_mask': padding_mask, 't': t, 'inds': style}

	def train_batch(self, batch):
		batch_dict = self.prepare_batch(batch)
		self.optimizer.zero_grad()
		y = self.model(batch_dict)
		loss = seq_ce_logits_loss(y, batch_dict['t'], batch_dict['lens'], batch_dict['enc_padding_mask'], True)
		acc = seq_acc(y, batch_dict['t'], batch_dict['lens'], batch_dict['enc_padding_mask'], True)
		loss_value = loss.item()
		ppl = math.exp(loss_value)
		loss.backward()
		clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
		self.optimizer.step()
		self.lr_scheduler.step()

		self.summary_writer.add_scalar('train/loss', loss_value, self.step)
		self.summary_writer.add_scalar('train/ppl', ppl, self.step)
		self.summary_writer.add_scalar('train/acc', acc, self.step)

		return loss_value, acc, ppl

	def train(self):
		self.model.train()
		data_iter = iter(self.train_loader)
		start = time.time()
		while self.step <= self.n_iters:
			# update_lr(self.optimizer, self.lr, self.step, self.update_interval, 
							# self.lr_up_start, self.lr_up_end, self.lr_down_start, self.lr_down_end, self.up_alpha, self.down_alpha, self.eps)
			batch = next(data_iter)
			loss, acc, ppl = self.train_batch(Batch(batch, self.train_loader.dataset, self.device, 
				ignore_fields=[] if self.diff_bias else ['label']))
			
			if self.step % self.log_interval == 0:
				print('step [{}/{}] loss: {:.4f}; acc: {:.2%}; ppl: {:.4f} | {:.2f} s elapsed'. format(
							self.step, self.n_iters, loss, acc, ppl, time.time() - start))
			if self.step % self.eval_interval == 0:
				valid_ppl = self.eval('valid', self.valid_loader)
				test_ppl = self.eval('test', self.test_loader)

				save_flag = False
				if valid_ppl < self.best_results['valid'].ppl:
					save_flag = True
					self.best_results['valid'] = Record(step = self.step, ppl = valid_ppl, the_other = test_ppl)
				if test_ppl < self.best_results['test'].ppl:
					save_flag = True
					self.best_results['test'] = Record(step = self.step, ppl = test_ppl, the_other = valid_ppl)
				print('current best valid: step {0.step} ppl {0.ppl:.4f} [{0.the_other:.4f}]'.format(self.best_results['valid']))
				print('current best test: step {0.step} ppl {0.ppl:.4f} [{0.the_other:.4f}]'.format(self.best_results['test']))
				if save_flag:
					self.save_states()
				self.save_states('latest-')
				# to save space
				if self.step != self.eval_interval:
					os.remove(os.path.join(self.model_path, 'latest-model-{}'.format(self.step - self.eval_interval)))
			self.step += 1
		self.summary_writer.close()

	def eval(self, name, dataset_loader):
		self.model.eval()
		n_total = len(dataset_loader.dataset)
		total_loss, total_acc = 0, 0
		start = time.time()
		with torch.no_grad():
			for batch in dataset_loader:
				batch_dict = self.prepare_batch(Batch(batch, dataset_loader.dataset, self.device, 
					ignore_fields=[] if self.diff_bias else ['label']))
				y = self.model(batch_dict)
				total_loss += seq_ce_logits_loss(y, batch_dict['t'], batch_dict['lens'], batch_dict['enc_padding_mask'], False).item()
				total_acc += seq_acc(y, batch_dict['t'], batch_dict['lens'], batch_dict['enc_padding_mask'], False)
		total_loss /= n_total
		total_acc /= n_total
		total_ppl = math.exp(total_loss)

		if not self.eval_only:
			self.summary_writer.add_scalar('{}/loss'.format(name), total_loss, self.step)
			self.summary_writer.add_scalar('{}/ppl'.format(name), total_ppl, self.step)
			self.summary_writer.add_scalar('{}/acc'.format(name), total_acc, self.step)

		print('{} performance of model at step {}'.format(name, self.step))
		print('loss: {:.4f}; acc: {:.2%}; ppl: {:.4f} | {:.2f} s for evaluation'. format(
							total_loss, total_acc, total_ppl, time.time() - start))
		self.model.train()
		return total_ppl

if __name__=='__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument('version', type=str)
	parser.add_argument('-seed', type=int, default=1)
	parser.add_argument('-gpu', type=int, default=0)
	parser.add_argument('-eval_only', type=str2bool, default=False)
	parser.add_argument('-save_only', type=str2bool, default=False)
	parser.add_argument('-ignore_splits', type=str2bool, default=False)
	parser.add_argument('-train_from', type=str, default=None, nargs='?')
	parser.add_argument('-select_label', type=int, default=None, nargs='?')
	parser.add_argument('-diff_bias', type=str2bool, default=False, nargs='?')
	parser.add_argument('-pretrained_emb', type=str, default=None, nargs='?')
	parser.add_argument('-pretrained_emb_max', type=int, default=None, nargs='?')
	parser.add_argument('-model_type', type=str, default='rnn', choices=['rnn', 'sa'])
	parser.add_argument('-reverse', type=str2bool, default=False)

	parser.add_argument('-work_dir', type=str, default='./')
	parser.add_argument('-dataset', type=str, default='yelp')
	parser.add_argument('-max_sen_len', type=int, default=None, nargs='?')
	parser.add_argument('-min_freq', type=int, default=1)
	parser.add_argument('-max_vocab_size', type=int, default=None, nargs='?')
	parser.add_argument('-batch_size', type=int, default=64)
	parser.add_argument('-eval_batch_size', type=int, default=64)

	parser.add_argument('-emb_size', type=int, default=200)
	parser.add_argument('-emb_max_norm', type=float, default=1.0, nargs='?')
	parser.add_argument('-hid_size', type=int, default=100)
	parser.add_argument('-rnn_type', type=str, default='GRU', nargs='?')
	parser.add_argument('-dropout_rate', type=float, default=0)
	parser.add_argument('-num_heads', type=int, default=4, nargs='?')
	parser.add_argument('-num_layers', type=int, default=4, nargs='?')
	parser.add_argument('-tie_weights', type=str2bool, default=False)
	parser.add_argument('-pos_sincode', type=str2bool, default=True, nargs='?')
	parser.add_argument('-token_emb_scale', type=str2bool, default=True, nargs='?')
	parser.add_argument('-att_dropout_rate', type=float, default=0, nargs='?')
	parser.add_argument('-transformer_norm_bf', type=str2bool, default=False, nargs='?')

	parser.add_argument('-max_grad_norm', type=float, default=2.0)
	parser.add_argument('-optim_method', type=str, default='adam')
	parser.add_argument('-momentum', type=float, default=None, nargs='?')
	parser.add_argument('-weight_decay', type=float, default=0)
	parser.add_argument('-lr', type=float, default=0.001)
	parser.add_argument('-beta2', type=float, default=0.999, nargs='?')
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

	config=parser.parse_args()
	print(' '.join(sys.argv))
	print(config)

	random.seed(config.seed)
	np.random.seed(config.seed+1)
	torch.manual_seed(config.seed+2)
	torch.cuda.manual_seed(config.seed+3)

	print('Start time: ', time.strftime('%X %x %Z'))
	trainer = Solver(config)
	if config.eval_only:
		trainer.eval('test', trainer.test_loader)
	elif config.save_only:
		trainer.save_states('full-')
	else:
		trainer.train()
	print('Finish time: ', time.strftime('%X %x %Z'))

