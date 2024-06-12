import numpy as np
import torch
import torchaudio
import h5py
import os

from typing import List


def mfcc_features(wav_path: str, config: dict) -> torch.Tensor:
	r"""
	Extract mfcc features from the audio wave.

	Args:
		wav_path (str): the wav path.
		config (dict): params setting for mfcc.
			containing the params below:

			- n_mfcc (int): Number of mfc coefficients to retain.
			- win_length (float): the length of a frame in the time domain, unit (s).
			- hop_length (float): the hop length in the time domain, unit (s).
			- n_mels (int): Number of mel filterbanks.

	Returns:
		The extracted mfcc features (torch.Tensor).
	"""
	wav, sr = torchaudio.load(wav_path)

	n_mfcc = config['n_mfcc']
	win_length = config['win_length']
	hop_length = config['hop_length']
	n_fft, hop_length = int(win_length * sr), int(hop_length * sr)

	melkwargs = {'n_fft': n_fft, 'hop_length': hop_length, 'n_mels': config['n_mels']}
	mfcc = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=n_mfcc, melkwargs=melkwargs)
	features = mfcc(wav)
	return features


def mel_features(wav_path: str, config: dict) -> torch.Tensor:
	r"""
	Extract mel-spectrogram features from the audio wave.

	Args:
		wav_path (str): the wav path.
		config (dict): params setting for mel-spectrogram.
			containing the params below:

			- win_length (float): the length of a frame in the time domain, unit (s).
			- hop_length (float): the hop length in the time domain, unit (s).
			- n_mels (int): Number of mel filterbanks.

	Returns:
		The extracted mel-spectrogram features (torch.Tensor).
	"""
	wav, sr = torchaudio.load(wav_path)

	win_length = config['win_length']
	hop_length = config['hop_length']
	n_mels = config['n_mels']
	n_fft, hop_length = int(win_length * sr), int(hop_length * sr)
	n_fft, hop_length = int(win_length * sr), int(hop_length * sr)
	mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
	features = mel(wav)
	return features


def get_label_features(label_dir: str, test_speakers: List[int], config: dict, feature_func_name: str = 'mfcc'):
	r""" get the all features with a certain label. """
	if len(test_speakers) == 0:
		raise ValueError('test speakers must be more than 0.')

	if feature_func_name == 'mfcc':
		feature_func = mfcc_features
	else:
		feature_func = mel_features

	audio_files = os.listdir(label_dir)
	train_features = []
	test_features = []
	train_lens = []
	test_lens = []
	for name in audio_files:
		speaker = int(name.split('-')[0])
		wav_path = os.path.join(label_dir, name)
		features = feature_func(wav_path=wav_path, config=config).squeeze(0)
		if speaker in test_speakers:
			test_features.append(features)
			test_lens.append(features.size()[1])
		else:
			train_features.append(features)
			train_lens.append(features.size()[1])

	print('In path: {}, train len: {}, test len: {}'.format(label_dir, len(train_features), len(test_features)))
	return train_features, test_features, train_lens, test_lens


def padding_features(tensor_list: List[torch.Tensor], max_len: int, padding: float = 0.):
	r""" pad the features with various length to meet the length of their max_len """
	padded_tensors = [torch.cat((tensor, torch.empty((*tensor.size()[:-1], max_len - tensor.size()[-1]),
													 dtype=tensor.dtype).fill_(padding)), dim=-1) for tensor in tensor_list]
	return padded_tensors


def normalize_features(tensor_list: List[torch.Tensor]):
	r""" normalize the features """
	tmp_tensor = torch.cat(tensor_list, dim=-1)
	std, mean = torch.std(tmp_tensor, dim=-1, keepdim=True), torch.mean(tmp_tensor, dim=-1, keepdim=True)
	normalized_tensors = [(tensor - mean) / std for tensor in tensor_list]
	return normalized_tensors


def save_features(save_path, data, labels, lens):
	r""" save features, labels, lengths of features """
	with h5py.File(save_path, 'w') as f:
		f.create_dataset(name='data', data=data)
		f.create_dataset(name='labels', data=labels)
		f.create_dataset(name='lens', data=lens)
	print('data saved to {}'.format(save_path))


def get_all_features(config: dict):
	r""" obtain all features """
	root = config['root']
	data_flag = config['config_flag']
	feature_func_name = config['feature_func_name']
	test_ratio = config['test_ratio']
	feature_func_name = 'mel-spectrogram' if feature_func_name != 'mfcc' else 'mfcc'
	save_dir = r'{}/data/{}'.format(root, feature_func_name)
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	train_data_path = r'{}/{}_train.h5'.format(save_dir, data_flag)
	test_data_path = r'{}/{}_test.h5'.format(save_dir, data_flag)

	if os.path.exists(train_data_path) and os.path.exists(test_data_path):
		return train_data_path, test_data_path

	segment_data_dir = r'{}/data/voices'.format(root)
	contents = os.listdir(segment_data_dir)

	speakers = np.loadtxt(os.path.join(segment_data_dir, 'speakers.txt')).astype(int)
	test_num = int(speakers.shape[0] * test_ratio)

	train_data, train_labels = [], []
	test_data, test_labels = [], []
	train_lens, test_lens = [], []
	for item in contents:
		if item.split('-')[0] != 'label':
			continue

		label = int(item.split('-')[-1])
		label_dir = os.path.join(segment_data_dir, item)
		test_speakers = list(np.random.choice(speakers, size=test_num, replace=False))
		results = get_label_features(label_dir=label_dir, test_speakers=test_speakers, config=config,
									 feature_func_name=feature_func_name)
		label_train_data, label_test_data, label_train_lens, label_test_lens = results

		train_lens += label_train_lens
		test_lens += label_test_lens
		train_data += label_train_data
		train_labels.append(torch.empty(len(label_train_data), dtype=torch.int).fill_(label))
		test_data += label_test_data
		test_labels.append(torch.empty(len(label_test_data), dtype=torch.int).fill_(label))

	train_data = normalize_features(train_data)
	test_data = normalize_features(test_data)

	train_data = padding_features(train_data, max_len=max(train_lens))
	test_data = padding_features(test_data, max_len=max(test_lens))

	train_data = torch.stack(train_data, dim=0).numpy()
	train_labels = torch.cat(train_labels, dim=0).numpy()
	test_data = torch.stack(test_data, dim=0).numpy()
	test_labels = torch.cat(test_labels, dim=0).numpy()

	train_lens = np.array(train_lens, dtype=int)
	test_lens = np.array(test_lens, dtype=int)

	save_features(save_path=train_data_path, data=train_data, labels=train_labels, lens=train_lens)
	save_features(save_path=test_data_path, data=test_data, labels=test_labels, lens=test_lens)
	return train_data_path, test_data_path
