import os
import torch
import numpy as np

from ..data_preprocess import save_audio_segments


def prepare_data():
	root = __file__
	for i in range(3):
		root = os.path.dirname(root)

	speakers_path = os.path.join(root, r'data/voices/speakers.txt')
	if not os.path.exists(speakers_path):
		save_audio_segments(raw_data_dir=os.path.join(root, r'data/board'))
		save_audio_segments(raw_data_dir=os.path.join(root, r'data/pc_single_ch'))


def set_device(device_name):
	if device_name[:4] == 'cuda' and torch.cuda.is_available():
		return torch.device(device_name)
	else:
		return torch.device('cpu')


def set_seed(seed: int):
	torch.manual_seed(seed)
	np.random.seed(seed)
