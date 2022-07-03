import six
import random
import functools
import numpy as np
from itertools import chain
from .utils import get_tokenizer
from . import constants

def _cut_list(x, n, cut_mode='first'):
	if cut_mode is None:
		cut_mode = 'first'
	if cut_mode == 'first':
		return x[:n]
	elif cut_mode == 'last':
		return x[-n:]
	elif cut_mode == 'random':
		p=random.random()
		return x[-n:] if p>0.5 else x[:n]
	else:
		raise ValueError('the provided cut_mode {} is not supported, only effective with: first, last, and random'.format(cut_mode))


class Preprocessor(object):
	def __init__(self, lower=False, cut_length=None, cut_mode=None,
				 tokenize=None, tokenizer_language='en', stop_words=None, reverse=False):
		super(Preprocessor, self).__init__()
		self.lower=lower
		self.cut_length=cut_length
		self.cut_mode=cut_mode
		self.tokenizer_args = (tokenize, tokenizer_language)
		self.tokenize = get_tokenizer(tokenize, tokenizer_language)
		self.reverse = reverse
		try:
			self.stop_words = set(stop_words) if stop_words is not None else None
		except TypeError:
			raise ValueError("Stop words must be convertible to a set")

	def __call__(self, x):
		"""Load a single example using this field, tokenizing if necessary.

		If the input is a Python 2 `str`, it will be converted to Unicode
		first. If `sequential=True`, it will be tokenized. Then the input
		will be optionally lowercased and passed to the user-provided
		`preprocessing` Pipeline."""
		if (six.PY2 and isinstance(x, six.string_types) and
				not isinstance(x, six.text_type)):
			x = six.text_type(x, encoding='utf-8')
		assert isinstance(x, six.text_type)
		x = self.tokenize(x.rstrip('\n'))
		if self.lower:
			# x = Pipeline(six.text_type.lower)(x)
			x = [six.text_type.lower(w) for w in x]
		if self.stop_words is not None:
			x = [w for w in x if w not in self.stop_words]
		if self.cut_length is not None:
			x = _cut_list(x, self.cut_length, self.cut_mode)
		if self.reverse:
			x.reverse()
		return x

# noise model from paper "Unsupervised Machine Translation Using Monolingual Corpora Only"
def noise_input(x, unk, word_drop=0.0, k=3):
	x = x[:]
	n = len(x)
	for i in range(n):
		if random.random() < word_drop:
			x[i] = unk

	# slight shuffle such that |sigma[i]-i| <= k
	sigma = (np.arange(n) + (k+1) * np.random.rand(n)).argsort()
	return [x[sigma[i]] for i in range(n)]

def noise_batch(batch, unk, word_drop=0.0, drop_as_unk=0.5, word_insert=0.0, k=3, insert_self=0.0):
	insert_prob = word_drop + word_insert
	new_batch = []
	for j in range(len(batch)):
		x = batch[j]
		new_x = []
		n = len(x)
		no_discard = n < 8
		for i in range(n):
			p = random.random()
			if p < word_drop:
				if no_discard or random.random() < drop_as_unk:
					new_x.append(unk)
			else:
				new_x.append(x[i])
				if p < insert_prob:
					tok = x[i] if random.random() < insert_self else batch[j-1][int(random.random()*len(batch[j-1]))]
					new_x.append(tok)

		# slight shuffle such that |sigma[i]-i| <= k
		x, n = new_x, len(new_x)
		assert n > 0
		sigma = (np.arange(n) + (k+1) * np.random.rand(n)).argsort()
		new_batch.append([x[sigma[i]] for i in range(n)])
	return new_batch

def prepare_basic(field, minibatch, with_init, with_eos, with_len, reverse):
	output = field.pad(minibatch, use_init=with_init, use_eos=with_eos, include_lengths=with_len, reverse=reverse)
	return output
def prepare_tensor_basic(field, minibatch):
	pad_value = 0 if minibatch[0].dtype==torch.float else constants.PAD_ID
	output = field.pad(minibatch, pad_value=pad_value)
	return output
def mask_start(end):
	if end == 0:
		return 0
	p = np.random.random()
	if p >= 0.8:
		return 0
	elif p >= 0.6:
		return end
	else:
		return np.random.randint(end)
def mask_word(w, vocab_size):

	p = np.random.random()
	if p >= 0.2:
		return constants.MSK_ID
	elif p >= 0.1:
		return np.random.randint(5, vocab_size)
	else:
		return w

def masking_span(num_mask, total_len, fix_span):
	span_lens = []
	temp_num_mask = num_mask
	while True:
		if fix_span is not None:
			span_len_one = fix_span
		else:
			span_len_one = min(max(1, np.random.geometric(p=0.2)), 5)

		if temp_num_mask >= span_len_one:
			span_lens.append(span_len_one)
			temp_num_mask -= span_len_one
		else:
			span_lens.append(temp_num_mask)
			break
	assert sum(span_lens) == num_mask

	np.random.shuffle(span_lens)
	num_to_insert = total_len - num_mask
	slot_to_insert = len(span_lens) + 1
	slot_insert_nums = []
	for _ in range(slot_to_insert):
		if num_to_insert == 0:
			slot_insert_nums.append(0)
		else:
			insert_num = np.random.randint(0, num_to_insert)
			slot_insert_nums.append(insert_num)
			num_to_insert -= insert_num
	if num_to_insert != 0:
		slot_insert_nums[-1] += num_to_insert

	slot_id_mask = []
	now_id = 0
	for i in range(len(span_lens)):
		now_id += slot_insert_nums[i]
		end_id = now_id + span_lens[i]
		id_mask = np.arange(now_id, end_id)
		slot_id_mask.append(id_mask)
		now_id = end_id

	# id_mask_list = list(chain.from_iterable(slot_id_mask))
	# for id_mask in slot_id_mask:
	# 	# p = np.random.random()
	# 	for id_ in id_mask:
	# 		id_mask_list.append(id_)
	return np.concatenate(slot_id_mask)

def mask_input(x, mask_rate, mask_consc, mask_span, mask_span_len, vocab_size):
	n = len(x) + 1 # to include an additional <eos> token
	num_mask = max(1, round(n * mask_rate)) if mask_rate is not None else np.random.randint(1, n+1)
	if mask_consc:
		start = mask_start(n - num_mask)
		mask_ids = np.arange(start, start+num_mask)
	elif mask_span:
		mask_ids = masking_span(num_mask, n, mask_span_len)
	else:
		mask_ids = np.arange(n)
		np.random.shuffle(mask_ids)
		mask_ids = mask_ids[:num_mask]
	x_in = x.copy()
	x_in.append(constants.EOS_ID)
	x_out = x_in.copy()
	for i in range(n):
		if i in mask_ids:
			x_in[i] = mask_word(x_in[i], vocab_size)
		else:
			x_out[i] = constants.PAD_ID
	return x_in, x_out

def prepare_trans(field, minibatch, noisy, noise_drop, with_init, with_eos):
	if noisy:
		noisy_minibatch = [noise_input(x, constants.UNK_ID, noise_drop) for x in minibatch]
		xc = field.pad(noisy_minibatch)
	else:
		xc = None
	x, lens = field.pad(minibatch, use_init=with_init, use_eos=with_eos, include_lengths=True)

	# y_out = field.pad(minibatch, use_eos=True)
	return xc, x, lens#, y_out

def prepare_trans_new(field, minibatch, noisy, noise_drop, noise_drop_as_unk, noise_insert, noise_insert_self, 
	mask_src_rate, mask_src_consc, mask_src_span, mask_src_span_len, mask_tgt, mask_tgt_consc, mask_tgt_span, mask_tgt_span_len):
	ret = {}
	if noisy:
		noisy_minibatch = noise_batch(minibatch, constants.UNK_ID, noise_drop, noise_drop_as_unk, noise_insert, insert_self=noise_insert_self)
		ret['xc'], ret['cenc_lens'] = field.pad(noisy_minibatch, use_eos=True, include_lengths=True)
	# else:
	# 	xc, lensc = None, None

	if mask_src_rate > 0:
		masked_batch = [mask_input(sample, mask_src_rate, mask_src_consc, mask_src_span, mask_src_span_len, len(field.vocab)) for sample in minibatch]
		mx_enc_in, mx_enc_out = list(zip(*masked_batch))
		ret['mx_enc_in'] = field.pad(mx_enc_in)
		ret['mx_enc_out'] = field.pad(mx_enc_out)
	
	if mask_tgt:
		masked_batch = [mask_input(sample, None, mask_tgt_consc, mask_tgt_span, mask_tgt_span_len, len(field.vocab)) for sample in minibatch]
		mx_dec_in, mx_dec_out = list(zip(*masked_batch))
		ret['mx_dec_in'] = field.pad(mx_dec_in)
		ret['mx_dec_out'] = field.pad(mx_dec_out)

	ret['x'], ret['enc_lens'] = field.pad(minibatch, use_eos=True, include_lengths=True)
	return ret

def prepare_conv_trans(field, minibatch, noisy, noise_drop, noise_drop_as_unk, noise_insert, pad_msk):
	if noisy:
		noisy_minibatch = noise_batch(minibatch, constants.UNK_ID, noise_drop, noise_drop_as_unk, noise_insert)
		xc, lensc = field.pad(noisy_minibatch, include_lengths=True)
	else:
		xc, lensc = None, None
	x, lens = field.pad(minibatch, use_eos=True, include_lengths=True, pad_value=constants.MSK_ID if pad_msk else constants.EOS_ID)
	return xc, lensc, x, lens


def prepare_eval(field, minibatch):
	# x = field.pad(minibatch)
	x, lens = field.pad(minibatch, use_init=True, use_eos=True, include_lengths=True)
	# y_out = field.pad(minibatch, use_eos=True)
	return x, lens

class Postprocessor(object):
	"""docstring for Postprocessor"""
	def __init__(self, mode, noisy=None, noise_drop=None, noise_drop_as_unk=None, noise_insert=None, 
		with_init=False, with_eos=False, with_len=True, reverse=False, pad_msk=True, noise_insert_self=None,
		mask_src_rate=0, mask_src_consc=None, mask_src_span=None, mask_src_span_len=None,
		mask_tgt=None, mask_tgt_consc=None, mask_tgt_span=None, mask_tgt_span_len=None):
		super(Postprocessor, self).__init__()
		self.mode = mode
		self.mask_src_rate = mask_src_rate 
		self.noisy = noisy
		self.mask_tgt = mask_tgt
		# self.noise_drop = noise_drop
		# self.require_clean = require_clean
		if mode == 'lm':
			self.post_func = functools.partial(prepare_basic, with_init=True, with_eos=True, with_len=True, reverse=reverse)
		# elif mode == 'dec':
		#     self.post_func = functools.partial(prepare_dec, with_input=with_input)
		elif mode == 'class':
			self.post_func = functools.partial(prepare_basic, with_init=False, with_eos=True, with_len=True, reverse=False)
		elif mode == 'trans':
			self.post_func = functools.partial(prepare_trans, noisy=noisy, noise_drop=noise_drop, with_init=False, with_eos=with_eos)
		elif mode == 'trans_new':
			self.post_func = functools.partial(prepare_trans_new, noisy=noisy, noise_drop=noise_drop, noise_drop_as_unk=noise_drop_as_unk, noise_insert=noise_insert, noise_insert_self=noise_insert_self,
				mask_src_rate=mask_src_rate, mask_src_consc=mask_src_consc, mask_src_span=mask_src_span, mask_src_span_len=mask_src_span_len,
				mask_tgt=mask_tgt, mask_tgt_consc=mask_tgt_consc, mask_tgt_span=mask_tgt_span, mask_tgt_span_len=mask_tgt_span_len)
		elif mode == 'ctrans':
			self.post_func = functools.partial(prepare_conv_trans, noisy=noisy, noise_drop=noise_drop, noise_drop_as_unk=noise_drop_as_unk, noise_insert=noise_insert, pad_msk=pad_msk)
		elif mode == 'basic':
			self.post_func = functools.partial(prepare_basic, with_init=with_init, with_eos=with_eos, with_len=with_len, reverse=reverse)
		elif mode == 'tensor_basic':
			self.post_func = prepare_tensor_basic
		elif mode == 'eval':
			self.post_func = prepare_eval
		# elif mode == 'simple':
		#     self.post_func = prepare_simple
		# elif mode == 'value':
		#     self.post_func = prepare_value
		# elif mode == 'usr':
		#     self.post_func = usr_func
		else:
			raise ValueError('unsupported mode for postprocessing function!')
	def turn_off_noise(self):
		assert self.mode == 'trans' or self.mode=='trans_new' or self.mode == 'ctrans'
		self.post_func.keywords['noisy'] = False
		self.noisy = False
	def to_eval_mode(self):
		assert self.mode == 'trans_new'
		self.post_func.keywords['noisy'] = False
		self.post_func.keywords['mask_src_rate'] = 0
		self.post_func.keywords['mask_tgt'] = False
	def to_train_mode(self):
		assert self.mode == 'trans_new'
		self.post_func.keywords['noisy'] = self.noisy
		self.post_func.keywords['mask_src_rate'] = self.mask_src_rate
		self.post_func.keywords['mask_tgt'] = self.mask_tgt
		

	def __call__(self, field, minibatch):
		return self.post_func(field, minibatch)