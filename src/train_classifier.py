import time
import argparse
import sys
import os
from collections import namedtuple
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from datalib.iterator import BucketIterator
from datalib.batch import Batch
from data_loader import Corpus
from train_utils import *
from loss import *
from layers import get_padding_mask_on_size
from classifiers import cnn_classifier, attn_classifier, Transformer_classifier
from torch.utils.tensorboard import SummaryWriter

Record = namedtuple('Record', 'step acc the_other', defaults = (0, 0, 0))

class Solver(object):
	"""docstring for Solver"""
	def __init__(self, config):
		super(Solver, self).__init__()
		# self.cnn = config.cnn
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

		datasets = Corpus.iters_dataset_simple(config.work_dir, config.dataset, 'class', config.max_sen_len,
							config.min_freq, config.max_vocab_size, batch_first=(config.model_type == 'cnn'))
		if config.sub < 1:
			old_len = len(datasets[0])
			random.shuffle(datasets[0].examples)
			new_len = int(old_len * config.sub)
			datasets[0].examples = datasets[0].examples[:new_len]
			print('cutting the training data from {} to {}'.format(old_len, len(datasets[0])))
		if config.ignore_splits:
			# for classifier training, we don't use the test split for style transfer as it is too small
			# so we split the training set to new dev and test
			datasets = datasets[0].split([0.8, 0.1, 0.1], True)
		self.train_loader, self.valid_loader, self.test_loader = BucketIterator.splits(datasets, 
							batch_sizes = [config.batch_size, config.eval_batch_size, config.eval_batch_size], 
							retain_order = False)


		self.model_path = os.path.join(config.work_dir, 'model', config.dataset, 'classifier', config.version)
		self.summary_path = os.path.join(config.work_dir, 'summary', config.dataset, 'classifier', config.version)
		makedirs(self.model_path)
		makedirs(self.summary_path)
		if not (self.eval_only or config.save_only):
			self.summary_writer = SummaryWriter(self.summary_path)

		vocab_size = len(self.train_loader.dataset.fields['text'].vocab)
		num_classes = len(self.train_loader.dataset.fields['label'].vocab)

		print('train size:', len(self.train_loader.dataset))
		print('valid size:', len(self.valid_loader.dataset))
		print('test size:', len(self.test_loader.dataset))
		print('vocab size:', vocab_size)
		print('number of classes:', num_classes)

		self.model_type = config.model_type

		if config.model_type == 'cnn':
			self.args = [vocab_size, config.emb_size, config.emb_max_norm, config.filter_sizes, config.n_filters, 
							config.leaky, config.conv_pad, config.dropout_rate, num_classes]
			self.model = cnn_classifier(vocab_size, config.emb_size, config.emb_max_norm, config.filter_sizes, config.n_filters, 
							config.leaky, config.conv_pad, config.dropout_rate, num_classes)
		elif config.model_type == 'rnn':
			self.args = [vocab_size, config.emb_size, config.emb_max_norm, config.hid_size, config.rnn_type,
							True, False, True, config.dropout_rate, num_classes]
			self.model = attn_classifier(vocab_size, config.emb_size, config.emb_max_norm, config.hid_size, config.rnn_type,
							True, False, True, config.dropout_rate, num_classes)
		else:
			self.args = [vocab_size, config.emb_size, config.emb_max_norm, config.pos_sincode, config.token_emb_scale, config.max_sen_len, 
				config.num_heads, config.hid_size, config.num_layers, config.dropout_rate, num_classes, config.with_cls, 
				config.att_dropout_rate, config.transformer_norm_bf]
			self.model = Transformer_classifier(vocab_size, config.emb_size, config.emb_max_norm, config.pos_sincode, config.token_emb_scale, config.max_sen_len, 
				config.num_heads, config.hid_size,
				config.num_layers, config.dropout_rate, num_classes, config.with_cls, config.att_dropout_rate, config.transformer_norm_bf)
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
		x, lens = batch.text
		t = batch.label
		if config.model_type == 'cnn':
			padding_mask = get_padding_mask_on_size(x.size(1), lens, 0, 1)
		elif config.model_type == 'rnn':
			padding_mask = get_padding_mask_on_size(x.size(0), lens)
		else:
			padding_mask = get_padding_mask_on_size(x.size(0), lens, 0, 1)

		x_key = 'x_b' if config.model_type == 'cnn' else 'x'
		padding_mask_key = 'padding_mask' if config.model_type == 'rnn' else 'padding_mask_b'
		
		return {x_key: x, padding_mask_key: padding_mask, 'lens': lens, 't': t}

	def train_batch(self, batch):
		batch_dict = self.prepare_batch(batch)
		self.optimizer.zero_grad()
		model_output = self.model(batch_dict)
		
		loss = F.cross_entropy(model_output, batch_dict['t'])
		acc = unit_acc(model_output, batch_dict['t'])
		loss.backward()
		clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
		self.optimizer.step()
		self.lr_scheduler.step()
		loss_value = loss.item()
		self.summary_writer.add_scalar('train/loss', loss_value, self.step)
		self.summary_writer.add_scalar('train/acc', acc, self.step)

		return loss_value, acc

	def train(self):
		self.model.train()
		data_iter = iter(self.train_loader)
		start = time.time()
		while self.step <= self.n_iters:
			# update_lr(self.optimizer, self.lr, self.step, self.update_interval, 
							# self.lr_up_start, self.lr_up_end, self.lr_down_start, self.lr_down_end, self.up_alpha, self.down_alpha, self.eps)
			batch = next(data_iter)
			loss, acc = self.train_batch(Batch(batch, self.train_loader.dataset, self.device))
			
			if self.step % self.log_interval == 0:
				print('step [{}/{}] loss: {:.4f}; acc: {:.2%} | {:.2f} s elapsed'. format(
							self.step, self.n_iters, loss, acc, time.time() - start))
			if self.step % self.eval_interval == 0:
				valid_acc = self.eval('valid', self.valid_loader)
				test_acc = self.eval('test', self.test_loader)

				save_flag = False
				if valid_acc > self.best_results['valid'].acc:
					save_flag = True
					self.best_results['valid'] = Record(step = self.step, acc = valid_acc, the_other = test_acc)
				if test_acc > self.best_results['test'].acc:
					save_flag = True
					self.best_results['test'] = Record(step = self.step, acc = test_acc, the_other = valid_acc)
				print('current best valid: step {0.step} acc {0.acc:.2%} [{0.the_other:.2%}]'.format(self.best_results['valid']))
				print('current best test: step {0.step} acc {0.acc:.2%} [{0.the_other:.2%}]'.format(self.best_results['test']))
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
				batch_dict = self.prepare_batch(Batch(batch, dataset_loader.dataset, self.device))
				model_output = self.model(batch_dict)
				
				total_loss += F.cross_entropy(model_output, batch_dict['t'], reduction = 'sum').item()
				total_acc += unit_acc(model_output, batch_dict['t'], False)
		total_loss /= n_total
		total_acc /= n_total
		if not self.eval_only:
			self.summary_writer.add_scalar('{}/loss'.format(name), total_loss, self.step)
			self.summary_writer.add_scalar('{}/acc'.format(name), total_acc, self.step)

		print('{} performance of model at step {}'.format(name, self.step))
		print('loss: {:.4f}; acc: {:.2%} | {:.2f} s for evaluation'. format(
							total_loss, total_acc, time.time() - start))
		self.model.train()
		return total_acc

if __name__=='__main__':

	parser=argparse.ArgumentParser()
	parser.add_argument('version', type=str)
	parser.add_argument('-seed', type=int, default=1)
	parser.add_argument('-gpu', type=int, default=0)
	parser.add_argument('-eval_only', type=str2bool, default=False)
	parser.add_argument('-save_only', type=str2bool, default=False)
	parser.add_argument('-ignore_splits', type=str2bool, default=False)
	parser.add_argument('-model_type', type=str, default='cnn', choices=['cnn', 'rnn', 'sa'])
	parser.add_argument('-train_from', type=str, default=None, nargs='?')
	parser.add_argument('-sub', type=float, default=1)
	parser.add_argument('-pretrained_emb', type=str, default=None, nargs='?')
	parser.add_argument('-pretrained_emb_max', type=int, default=None, nargs='?')
	
	parser.add_argument('-work_dir', type=str, default='./')
	parser.add_argument('-dataset', type=str, default='yelp')
	parser.add_argument('-max_sen_len', type=int, default=None, nargs='?')
	parser.add_argument('-min_freq', type=int, default=1)
	parser.add_argument('-max_vocab_size', type=int, default=None, nargs='?')
	parser.add_argument('-batch_size', type=int, default=64)
	parser.add_argument('-eval_batch_size', type=int, default=64)

	parser.add_argument('-emb_size', type=int, default=200)
	parser.add_argument('-emb_max_norm', type=float, default=1.0, nargs='?')
	parser.add_argument('-filter_sizes', type=str2intlist, default=[3,4,5], nargs='?')
	parser.add_argument('-n_filters', type=int, default=100, nargs='?')
	parser.add_argument('-leaky', type=str2bool, default=False, nargs='?')
	parser.add_argument('-conv_pad', type=str2bool, default=False, nargs='?')
	parser.add_argument('-hid_size', type=int, default=100, nargs='?')
	parser.add_argument('-rnn_type', type=str, default='GRU', nargs='?')
	parser.add_argument('-num_heads', type=int, default=4, nargs='?')
	parser.add_argument('-num_layers', type=int, default=4, nargs='?')
	parser.add_argument('-with_cls', type=str2bool, default=False, nargs='?')
	parser.add_argument('-dropout_rate', type=float, default=0)
	parser.add_argument('-att_dropout_rate', type=float, default=0, nargs='?')
	parser.add_argument('-transformer_norm_bf', type=str2bool, default=False, nargs='?')
	parser.add_argument('-pos_sincode', type=str2bool, default=True, nargs='?')
	parser.add_argument('-token_emb_scale', type=str2bool, default=True, nargs='?')

	parser.add_argument('-max_grad_norm', type=float, default=2.0)
	parser.add_argument('-optim_method', type=str, default='adam')
	parser.add_argument('-momentum', type=float, default=None, nargs='?')
	parser.add_argument('-weight_decay', type=float, default=0)
	parser.add_argument('-beta2', type=float, default=0.999, nargs='?')
	parser.add_argument('-lr', type=float, default=0.001)
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