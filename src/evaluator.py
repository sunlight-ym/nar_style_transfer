import sys
import math
import argparse
import time
import torch
import codecs
from collections import namedtuple
from datalib.iterator import BucketIterator
from datalib.batch import Batch
import datalib.constants as constants
from data_loader import *
from train_utils import *
from loss import *
from classifiers import *
from lms import *
from layers import get_padding_mask_on_size, reverse_seq
from models import style_transfer, style_transfer_transformer

Record = namedtuple('Record', 'a1 a2 a3')
STRecord = namedtuple('STRecord', 'a1 a2 a3 a4 a5')

class Evaluator(object):
	"""docstring for Evaluator"""
	def __init__(self, config):
		super(Evaluator, self).__init__()
		self.bin_size = config.bin_size
		self.num_bins = config.num_bins
		self.batch_size = config.batch_size
		self.num_ref = config.num_ref
		self.refine_passes = config.refine_passes
		self.trans_extra_len = config.trans_extra_len
		self.less_len = config.less_len
		self.device = torch.device('cuda', config.gpu)
		testset = Corpus.iters_output(config.work_dir, config.dataset, config.model_name, config.text_vocab, config.label_vocab, model_is_file_path=config.model_is_file_path)
		print('performance of model {} on dataset {}'.format(config.model_name, config.dataset))
		self.test_loader = BucketIterator(testset, batch_size=config.batch_size, train=False)
		# self.true_sens = read_sens(get_human_file(config.work_dir, config.dataset)) if config.with_truth else None
		# self.src_sens = read_sens(get_model_src_file(config.work_dir, config.dataset, config.model_out_dir))
		if self.refine_passes > 0:
			refine_path = os.path.join(config.work_dir, 'refine', config.dataset)
			makedirs(refine_path)
			identifier = f'M{config.model_name.replace("/", "_")}_R{config.refine_tool.replace("/", "_")}_N{self.refine_passes}'
			self.refine_path = os.path.join(refine_path, identifier)
			self.itos = testset.fields['text'].vocab.itos
			self.trans_sens = []

		print('test size:', len(self.test_loader.dataset))
		# print('src input size:', len(self.src_sens))
		# print('true output size:', len(self.true_sens) if self.true_sens is not None else 0)
		self.num_classes = len(testset.fields['label'].vocab)
		self.human_files = get_human_files(config.work_dir, config.dataset, self.num_classes, config.num_ref)
		self.test_files = get_test_files(config.work_dir, config.dataset, self.num_classes)
		if config.model_is_file_path:
			self.model_out_files = get_model_files_custom(config.model_name, self.num_classes)
		else:
			self.model_out_files = get_model_files(config.work_dir, config.dataset, config.model_name, self.num_classes)
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

		if self.refine_passes > 0:
			check_point = torch.load(config.refine_tool, map_location=lambda storage, loc: storage)
			self.use_transformer = check_point['use_transformer']
			tool_type = style_transfer_transformer if self.use_transformer else style_transfer
			# tool_type = {'rnn': rnn_lm, 'sa': Transformer_lm}[tool_type]
			self.refine_tool = tool_type(*check_point['args'])
			self.refine_tool.to(self.device)
			self.refine_tool.load_state_dict(check_point['model_state'])
			del check_point
			frozen_model(self.refine_tool)
			self.refine_tool.eval()


	def prepare_batch(self, batch):
		full_text, lens = batch.text
		lens = lens - 1
		tgt_style = batch.label
		if self.refine_passes > 0:
			with torch.no_grad():
				enc_text, enc_lens, enc_mask = full_text[1:], lens, None
				bsz = tgt_style.size(0)
				for i in range(self.refine_passes):
					refine_batch_dict = {'x': enc_text, 'enc_lens': enc_lens, 'style': tgt_style}

					if self.use_transformer:
						refine_batch_dict['seq_mask_b'] = get_padding_mask_on_size(enc_text.size(0), enc_lens, 0, 1) if enc_mask is None else enc_mask
					else:
						refine_batch_dict['seq_mask'] = get_padding_mask_on_size(enc_text.size(0), enc_lens) if enc_mask is None else enc_mask
					refine_result = self.refine_tool.transfer(refine_batch_dict, False, False, True, False, False, None, 1, True, 
						self.trans_extra_len, self.less_len, same_target=True)[1]['fw']
					max_out_len = refine_result['hard_outputs_lens_with_eos'].max().item()
					enc_text = refine_result['hard_outputs'][:max_out_len]
					enc_lens = refine_result['hard_outputs_lens_with_eos']
					enc_mask = refine_result['hard_outputs_padding_mask_t_with_eos' if self.use_transformer else 
						'hard_outputs_padding_mask_with_eos'][:max_out_len]
					
				to_sentences(enc_text, enc_lens-1, self.itos, self.trans_sens)
				full_text = torch.cat([enc_text.new_full((1, bsz), constants.BOS_ID), enc_text], 0)
				lens = enc_lens




		x_lm, y_lm = full_text[:-1], full_text[1:]
		x_class = full_text[1:]
		class_batch_dict = {'t': tgt_style}
		padding_mask = get_padding_mask_on_size(full_text.size(0)-1, lens)
		if isinstance(self.style_eval_tool, cnn_classifier):
			class_batch_dict['x_b'] = x_class.t()
			class_batch_dict['padding_mask_b'] = padding_mask.t()
		elif isinstance(self.style_eval_tool, attn_classifier):
			class_batch_dict['x'] = x_class
			class_batch_dict['padding_mask'] = padding_mask
		else:
			class_batch_dict['x'] = x_class
			class_batch_dict['padding_mask_b'] = padding_mask.t()
			class_batch_dict['lens'] = lens
		lm_batch_dict = {'x': x_lm, 'inds': tgt_style, 'y': y_lm, 'lens': lens, 'padding_mask': padding_mask}
		if self.bi_fluency_eval:
			x_lm_r = torch.cat([x_lm[0:1], reverse_seq(x_lm[1:], lens-1)], 0)
			y_lm_r = reverse_seq(y_lm, lens-1, y_lm==constants.EOS_ID)
			lm_rev_batch_dict = {'x': x_lm_r, 'inds': tgt_style, 'y': y_lm_r, 'lens': lens, 'padding_mask': padding_mask}
		else:
			lm_rev_batch_dict = None


		return class_batch_dict, lm_batch_dict, lm_rev_batch_dict

	def save_refine_results(self):
		self.trans_sens = reorder(self.test_loader.order, self.trans_sens)
		src_sens = []
		num_lines = []
		for test_file in self.test_files:
			with open(test_file, 'r') as ft:
				lines = ft.readlines()
				src_sens.extend(lines)
				num_lines.append(len(lines))
		with open(self.refine_path, 'w') as f:
			for src, tgt in zip(src_sens, self.trans_sens):
				f.write(src.strip() + '\t' + ' '.join(tgt) + '\n')
		new_out_files = []
		k = 0
		for i in range(self.num_classes):
			out_file = f'{self.refine_path}.{i}'
			new_out_files.append(out_file)
			with open(out_file, 'w') as f:
				for j in range(num_lines[i]):
					f.write(' '.join(self.trans_sens[k]) + '\n')
					k += 1
		return new_out_files

	def eval(self):
		n_total = len(self.test_loader.dataset)
		total_acc, total_nll = 0, 0
		start = time.time()
		with torch.no_grad():
			for batch in self.test_loader:
				class_batch_dict, lm_batch_dict, lm_rev_batch_dict = self.prepare_batch(Batch(batch, self.test_loader.dataset, self.device))
				style_eval_logits = self.style_eval_tool(class_batch_dict)
				total_acc += unit_acc(style_eval_logits, class_batch_dict['t'], False)

				fluency_eval_logits = self.fluency_eval_tool(lm_batch_dict)
				fluency_nll = seq_ce_logits_loss(fluency_eval_logits, lm_batch_dict['y'], lm_batch_dict['lens'], lm_batch_dict['padding_mask'], False).item()
				if self.bi_fluency_eval:
					fluency_eval_logits_rev = self.fluency_eval_tool_rev(lm_rev_batch_dict)
					fluency_nll += seq_ce_logits_loss(fluency_eval_logits_rev, lm_rev_batch_dict['y'], lm_rev_batch_dict['lens'], lm_rev_batch_dict['padding_mask'], False).item()
					fluency_nll /= 2
				total_nll += fluency_nll

		total_acc /= n_total
		total_nll /= n_total
		total_ppl = math.exp(total_nll)
		# trans_sens = list(self.test_loader.dataset.text)
		# self_bleu = get_bleu(trans_sens, self.src_sens)
		# human_bleu = get_bleu(trans_sens, self.true_sens) if self.true_sens is not None else None

		if self.refine_passes > 0:
			self.model_out_files = self.save_refine_results()

		self_bleu = 0
		human_bleu = 0
		for i in range(self.num_classes):
			self_bleu += compute_bleu_score(self.test_files[i], self.model_out_files[i])
			human_bleu += compute_bleu_score(self.human_files[i], self.model_out_files[i])
		self_bleu /= self.num_classes
		human_bleu /= self.num_classes

		print('transfer accuracy: {:.2%}\n'.format(total_acc))
		print('transfer perplexity: {:.2f}\n'.format(total_ppl))
		print('self bleu: {:.2f}\n'.format(self_bleu))
		print('human bleu: {:.2f}\n'.format(human_bleu))

		if self.refine_passes > 0:

			# self.save_refine_results()
			append_scores(self.refine_path, total_acc, total_ppl, self_bleu, human_bleu)

		print('{:.2f} s for evaluation'.format(time.time() - start))



	def bleu_cal_for_bin(self, src, gen, tgt_list, bin_id):
		with open(f'src_bin_{bin_id}', 'w') as f:
			for line in src:
				f.write(line)
		with open(f'gen_bin_{bin_id}', 'w') as f:
			for line in gen:
				f.write(line)
		for i in range(self.num_ref):
			with open(f'tgt_bin_{bin_id}_{i}', 'w') as f:
				for line in tgt_list[i]:
					f.write(line)

		self_bleu = compute_bleu_score(f'src_bin_{bin_id}', f'gen_bin_{bin_id}')
		human_bleu = compute_bleu_score([f'tgt_bin_{bin_id}_{i}' for i in range(self.num_ref)], f'gen_bin_{bin_id}')
		os.remove(f'src_bin_{bin_id}')
		os.remove(f'gen_bin_{bin_id}')
		for i in range(self.num_ref):
			os.remove(f'tgt_bin_{bin_id}_{i}')
		return self_bleu, human_bleu

	


	def eval_bins(self):
		n_total = len(self.test_loader.dataset)
		# total_acc, total_nll = 0, 0
		start = time.time()

		acc_stat = torch.zeros(self.num_bins, dtype=torch.float, device=self.device)
		nll_stat = torch.zeros(self.num_bins, dtype=torch.float, device=self.device)
		self_bleu_stat = torch.empty(self.num_bins, dtype=torch.float)
		human_bleu_stat = torch.empty(self.num_bins, dtype=torch.float)
		bin_capacity = torch.zeros(self.num_bins, dtype=torch.float, device=self.device)

		record_bins_test = [[] for i in range(self.num_bins)]
		record_bins_model = [[] for i in range(self.num_bins)]
		record_bins_human = [[[] for k in range(self.num_ref)] for i in range(self.num_bins)]
		ind_list = torch.empty(n_total, dtype=torch.long, device=self.device)
		j = 0
		max_len = 0
		for test_file, model_out, human_file_list in zip(self.test_files, self.model_out_files, self.human_files):
			with open(test_file, 'r') as ft:
				with codecs.open(model_out, 'r', encoding='utf-8', errors='ignore') as fm:
					fhs = [codecs.open(ref, 'r', encoding='utf-8', errors='ignore') for ref in human_file_list]
					for linet in ft:
						length = len(linet.strip().split(' '))
						if length > max_len:
							max_len = length
						ind = min(length//self.bin_size, self.num_bins-1)
						ind_list[j] = ind
						j += 1
						bin_capacity[ind] += 1
						record_bins_test[ind].append(linet)
						record_bins_model[ind].append(fm.readline())
						for k in range(self.num_ref):
							record_bins_human[ind][k].append(fhs[k].readline())
					for fh in fhs:
						fh.close()

		for i in range(self.num_bins):
			self_bleu_stat[i], human_bleu_stat[i] = self.bleu_cal_for_bin(record_bins_test[i], record_bins_model[i], record_bins_human[i], i)
		del record_bins_test, record_bins_model, record_bins_human
		
		assert bin_capacity.sum().item() == n_total





		with torch.no_grad():
			for batch_id, batch in enumerate(self.test_loader):
				if batch_id == 0:
					ind_list = ind_list[torch.tensor(self.test_loader.order, dtype=torch.long, device=self.device)]
				class_batch_dict, lm_batch_dict, lm_rev_batch_dict = self.prepare_batch(Batch(batch, self.test_loader.dataset, self.device))
				style_eval_logits = self.style_eval_tool(class_batch_dict)
				batch_acc = unit_acc(style_eval_logits, class_batch_dict['t'], False, False)

				fluency_eval_logits = self.fluency_eval_tool(lm_batch_dict)
				batch_nll = seq_ce_logits_loss(fluency_eval_logits, lm_batch_dict['y'], lm_batch_dict['lens'], lm_batch_dict['padding_mask'], False, reduction=False)
				if self.bi_fluency_eval:
					fluency_eval_logits_rev = self.fluency_eval_tool_rev(lm_rev_batch_dict)
					batch_nll += seq_ce_logits_loss(fluency_eval_logits_rev, lm_rev_batch_dict['y'], lm_rev_batch_dict['lens'], lm_rev_batch_dict['padding_mask'], False, reduction=False)
					batch_nll /= 2
				for bin_id, acc, nll in zip(ind_list[(batch_id*self.batch_size):(batch_id*self.batch_size+batch_acc.size(0))], batch_acc, batch_nll):
					acc_stat[bin_id] += acc
					nll_stat[bin_id] += nll

		acc_stat /= bin_capacity
		nll_stat /= bin_capacity
		ppl_stat = torch.exp(nll_stat)
		# trans_sens = list(self.test_loader.dataset.text)
		# self_bleu = get_bleu(trans_sens, self.src_sens)
		# human_bleu = get_bleu(trans_sens, self.true_sens) if self.true_sens is not None else None

		
		print('max length', max_len)
		bounds = [f'[{i*self.bin_size},{(i+1)*self.bin_size})' for i in range(self.num_bins-1)]
		bounds.append(f'[{(self.num_bins-1)*self.bin_size},inf)')
		print('bins:\t', bounds)
		print('bin capacity:\t', bin_capacity)

		print('transfer accuracy:\t', acc_stat)
		print('transfer perplexity:\t', ppl_stat)
		print('self bleu:\t', self_bleu_stat)
		print('human bleu:\t', human_bleu_stat)
		print('{:.2f} s for evaluation'.format(time.time() - start))


if __name__=='__main__':

	parser=argparse.ArgumentParser()
	parser.add_argument('-gpu', type=int, default=0)
	parser.add_argument('-work_dir', type=str, default='./')
	parser.add_argument('-dataset', type=str, default='yelp')
	parser.add_argument('-model_name', type=str, default='cae')
	parser.add_argument('-num_ref', type=int, default=1)
	parser.add_argument('-text_vocab', type=str, default=None)
	parser.add_argument('-label_vocab', type=str, default=None)
	
	parser.add_argument('-batch_size', type=int, default=64)

	parser.add_argument('-style_eval_tool', type=str, default=None)
	parser.add_argument('-fluency_eval_tool', type=str, default=None)
	parser.add_argument('-fluency_eval_tool_rev', type=str, default=None, nargs='?')
	
	parser.add_argument('-refine_tool', type=str, default=None, nargs='?')
	parser.add_argument('-refine_passes', type=int, default=0)
	parser.add_argument('-less_len', type=int, default=None, nargs='?')
	parser.add_argument('-trans_extra_len', type=int, default=0, nargs='?')

	parser.add_argument('-model_is_file_path', type=str2bool, default=False)
	parser.add_argument('-eval_by_len', type=str2bool, default=False)
	parser.add_argument('-bin_size', type=int, default=2, nargs='?')
	parser.add_argument('-num_bins', type=int, default=5, nargs='?')

	config=parser.parse_args()
	print(' '.join(sys.argv))
	print(config)

	print('Start time: ', time.strftime('%X %x %Z'))
	evaluator = Evaluator(config)
	if config.eval_by_len:
		evaluator.eval_bins()
	else:
		evaluator.eval()

	print('Finish time: ', time.strftime('%X %x %Z'))
