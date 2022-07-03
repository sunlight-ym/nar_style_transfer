import os
from collections import OrderedDict
import dill
import random
import math
from itertools import chain
import codecs
import torch
from datalib.dataset import Dataset
from datalib.example import Example
from datalib.field import Field, LabelField, NumField
from datalib.processor import Preprocessor, Postprocessor
import pickle
import torch.nn.functional as F
from train_utils import makedirs


def make_vocab(vocab_file, field, srcs, min_freq, max_vocab_size):
	if os.path.exists(vocab_file):
		with open(vocab_file, 'rb') as f:
			field.vocab = dill.load(f)
		print('finish loading the vocab file %s '%(vocab_file))
	else:
		field.build_vocab(srcs, min_freq=min_freq, max_size=max_vocab_size)
		with open(vocab_file, 'wb') as f:
			dill.dump(field.vocab, f)
		print('finish creating the vocab file %s '%(vocab_file))

def get_data(path, name):
	data_file = os.path.join(path, 'datasets', name, 'processed', '{}.pkl'.format(name))
	with open(data_file, 'rb') as f:
		datasets = pickle.load(f)
	
	return datasets['train'], datasets['dev'], datasets['test']
	
def get_human_files(path, name, num_class, num_ref):
	prefix = os.path.join(path, 'datasets', name, 'references', 'reference')
	files = []
	for i in range(num_class):
		files.append(['{}{}.{}'.format(prefix, j, i) for j in range(num_ref)])
	return files

def get_test_files(path, name, num_class):
	prefix = os.path.join(path, 'datasets', name, 'corpus', 'test')
	files = []
	for i in range(num_class):
		files.append('{}.{}'.format(prefix, i))
	return files
def get_test_file_lines(path, name, num_class):
	prefix = os.path.join(path, 'datasets', name, 'corpus', 'test')
	num_lines = []
	for i in range(num_class):
		with open('{}.{}'.format(prefix, i), 'r') as f:
			num_lines.append(len(f.readlines()))
	return num_lines

def get_model_output(path, name, model):
	data_file = os.path.join(path, 'datasets', name, 'model_out_processed', '{}.pkl'.format(model))
	with open(data_file, 'rb') as f:
		model_ouput = pickle.load(f)
	return model_ouput

def get_model_output_custom(path, name, file_path):
	data = []
	label_dict = ['<NEG>', '<POS>'] if name == 'yelp' or name == 'amazon' else ['<FORMAL>', '<INFORMAL>']
	if os.path.isfile(f'{file_path}.0'):
		for i in range(2):
			label = label_dict[1-i]
			with open(f'{file_path}.{i}', 'r') as f:
				for line in f:
					data.append({'text':line.strip(), 'label':label})
	else:
		num_lines = get_test_file_lines(path, name, 2)
		with open(file_path, 'r') as f:
			for i in range(2):
				label = label_dict[1-i]
				with open(f'{file_path}.{i}', 'w') as fo:
					for j in range(num_lines[i]):
						line = f.readline().split('\t')[1]
						data.append({'text':line.strip(), 'label':label})
						fo.write(line)
	return data

def get_model_files(path, name, model, num_class):
	prefix = os.path.join(path, 'datasets', name, 'model_out', model, 'test')
	files = []
	for i in range(num_class):
		files.append('{}.{}.tsf'.format(prefix, i))
	return files

def get_model_files_custom(file_path_prefix, num_class):
	files = [f'{file_path_prefix}.{i}' for i in range(num_class)]
	return files

def get_model_files_custom2(path, name, file_path_prefix, num_class):
	files = [f'{file_path_prefix}.{i}' for i in range(num_class)]
	if not os.path.isfile(files[0]):
		num_lines = get_test_file_lines(path, name, num_class)
		with codecs.open(file_path_prefix, 'r', encoding='utf-8', errors='ignore') as f:
			for i in range(num_class):
				# label = label_dict[1-i]
				with open(f'{file_path_prefix}.{i}', 'w') as fo:
					for j in range(num_lines[i]):
						line = f.readline().split('\t')[1]
						# data.append({'text':line.strip(), 'label':label})
						fo.write(line)
	return files

# def get_model_src_file(path, name, model_out_dir):
# 	return os.path.join(path, 'datasets', name, model_out_dir, 'src_text')

def get_class_stats(dataset, field_name):
	# num_class = len(dataset.fields[field_name].vocab)
	stat = OrderedDict()
	for x in dataset:
		v = getattr(x, field_name)
		if v in stat:
			stat[v] += 1
		else:
			stat[v] = 1

	return list(stat.values())

class Corpus(Dataset):
	"""docstring for Corpus"""
	def __init__(self, data, fields, filter_pred=None, sort_key=None):
		examples = [Example.fromflatdict(x, fields) for x in data]
		# fields = filter(lambda t: t[1] is not None, fields)
		super(Corpus, self).__init__(examples, fields, filter_pred, sort_key)


	@classmethod
	def iters_dataset_simple(cls, path, name, mode, max_sen_len, 
						min_freq, max_vocab_size, 
						noisy=None, noise_drop=None, noise_drop_as_unk=None, noise_insert=None, noise_insert_self=None, 
						mask_src_rate=None, mask_src_consc=None, mask_src_span=None, mask_src_span_len=None,
						x_mask_tgt=None, mask_tgt_consc=None, mask_tgt_span=None, mask_tgt_span_len=None, px_mask_target=None,
						reverse=False, require_px_top=None,
						batch_first=False, with_pseudo=False, pseudo_path=None):
		# , batch_size, eval_batch_size, drop_last, label_split, select_label, device):

		dataset_list = get_data(path, name)

		vocab_path = os.path.join(path, 'vocabs')
		makedirs(vocab_path)
		tv_suffix = 'm{}_f{}_s{}'.format(max_sen_len, min_freq, max_vocab_size)
		text_vocab_file = os.path.join(vocab_path, '{}_text_{}.pkl'.format(name, tv_suffix))
		label_vocab_file = os.path.join(vocab_path, '{}_label_{}.pkl'.format(name, max_sen_len))

		add_toks = ['<MASK>']

		text_field = Field(preprocessing=Preprocessor(), 
			postprocessing=Postprocessor(mode, noisy, noise_drop, noise_drop_as_unk, noise_insert, noise_insert_self=noise_insert_self,
			mask_src_rate=mask_src_rate, mask_src_consc=mask_src_consc, mask_src_span=mask_src_span, mask_src_span_len=mask_src_span_len,
			mask_tgt=x_mask_tgt, mask_tgt_consc=mask_tgt_consc, mask_tgt_span=mask_tgt_span, mask_tgt_span_len=mask_tgt_span_len, 
			reverse=reverse), 
			batch_first=batch_first, additional_special_tokens=add_toks)
		label_field = LabelField()
		fields = {'text': text_field, 'label': label_field}

		filter_pred = (lambda ex: len(ex.text) <= max_sen_len and len(ex.text) > 0) if max_sen_len is not None else (lambda ex: len(ex.text) > 0)
		sort_key = lambda ex: len(ex.text)

		if with_pseudo:
			pseudo_field = Field(preprocessing=Preprocessor(), postprocessing=Postprocessor('trans_new',
			mask_tgt=px_mask_target, mask_tgt_consc=mask_tgt_consc, mask_tgt_span=mask_tgt_span, mask_tgt_span_len=mask_tgt_span_len), 
				batch_first=batch_first, additional_special_tokens=add_toks)
			train_fields = {'text': text_field, 'label': label_field, 'pseudo': pseudo_field}
			if require_px_top:
				topp_field = TensorField(postprocessing=Postprocessor('tensor_basic'), dtype=torch.float, batch_first=batch_first)
				topi_field = TensorField(postprocessing=Postprocessor('tensor_basic'), batch_first=batch_first)
				train_fields['topp'] = topp_field
				train_fields['topi'] = topi_field
			with open(pseudo_path, 'rb') as f:
				trainset = pickle.load(f)
			trainset = cls(trainset, train_fields, filter_pred, sort_key)
			dataset_list = [trainset] + [cls(d, fields, filter_pred, sort_key) if d is not None else None for d in dataset_list[1:]]
		else:
			dataset_list = [cls(d, fields, filter_pred, sort_key) if d is not None else None for d in dataset_list]

		
		# train, valid, test = cls.splits(path, name, fields=fields, filter_pred=filter_pred, sort_key=sort_key)
		make_vocab(text_vocab_file, text_field, dataset_list[0], min_freq, max_vocab_size)
		make_vocab(label_vocab_file, label_field, dataset_list[0], 1, None)
		if with_pseudo:
			pseudo_field.vocab = text_field.vocab
		
		
		return dataset_list
	

	# @classmethod
	# def iters_output(cls, path, name, model, text_vocab_file, label_vocab_file, batch_first=False):
	# 	text_field = Field(preprocessing=Preprocessor(), postprocessing=Postprocessor('eval'), batch_first=batch_first)
	# 	label_field = LabelField()
	# 	text_vocab_file = os.path.join(path, 'vocabs', text_vocab_file)
	# 	label_vocab_file = os.path.join(path, 'vocabs', label_vocab_file)
	# 	with open(text_vocab_file, 'rb') as f:
	# 		text_field.vocab = dill.load(f)
	# 	with open(label_vocab_file, 'rb') as f:
	# 		label_field.vocab = dill.load(f)
		
	# 	fields = {'text': text_field, 'label': label_field}
	# 	sort_key = (lambda ex: len(ex.text))
	# 	output = get_model_output(path, name, model)
	# 	output = cls(output, fields=fields, sort_key=sort_key)

	# 	return output

	@classmethod
	def iters_output(cls, path, name, model, text_vocab_file, label_vocab_file, batch_first=False, model_is_file_path=False):
		text_field = Field(preprocessing=Preprocessor(), postprocessing=Postprocessor('eval'), batch_first=batch_first)
		label_field = LabelField()
		text_vocab_file = os.path.join(path, 'vocabs', text_vocab_file)
		label_vocab_file = os.path.join(path, 'vocabs', label_vocab_file)
		with open(text_vocab_file, 'rb') as f:
			text_field.vocab = dill.load(f)
		with open(label_vocab_file, 'rb') as f:
			label_field.vocab = dill.load(f)
		
		fields = {'text': text_field, 'label': label_field}
		sort_key = (lambda ex: len(ex.text))
		output = get_model_output_custom(path, name, model) if model_is_file_path else get_model_output(path, name, model)
		output = cls(output, fields=fields, sort_key=sort_key)

		return output

		

def save_datasets(save_data_path, dataset_list):
	with open(save_data_path, 'wb') as f:
		data_serialization={}
		data_serialization['fields'] = dataset_list[0].fields
		data_serialization['sort_key'] = dataset_list[0].sort_key
		data_serialization['examples'] = [d.examples if d is not None else None for d in dataset_list]
		dill.dump(data_serialization, f)

def load_datasets(load_data_path):
	with open(load_data_path, 'rb') as f:
		data_serialization = dill.load(f)
	fields = data_serialization['fields']
	sort_key = data_serialization['sort_key']
	examples_list = data_serialization['examples']
	dataset_list = [Dataset(examples, fields, sort_key=sort_key) if examples is not None else None for examples in examples_list]
	return dataset_list