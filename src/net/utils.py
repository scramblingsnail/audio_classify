import torch
import os
import numpy as np


contents_map = {'weight_ih_l0': ['w_ii', 'w_if', 'w_ig', 'w_io'],
				'weight_hh_l0': ['w_hi', 'w_hf', 'w_hg', 'w_ho'],
				'bias_ih_l0': ['b_ii', 'b_if', 'b_ig', 'b_io'],
				'bias_hh_l0': ['b_hi', 'b_hf', 'b_hg', 'b_ho']}


def save_single_c_param(p_name: str, parameter: torch.Tensor, save_dir: str):
	r"""
	save each single parameter to txt.
	each weight matrix is of size (output_size, input_size)
	"""
	prefix, p_name = p_name.split('.')
	parameter = parameter.detach().numpy()
	if prefix == 'lstm':
		assert p_name in contents_map.keys()
		p_num = len(contents_map[p_name])
		assert parameter.shape[0] % p_num == 0

		hidden_size = parameter.shape[0] // p_num

		for idx in range(p_num):
			sub_p = parameter[idx * hidden_size: (idx + 1) * hidden_size]
			sub_name = contents_map[p_name][idx]
			# print(sub_name, sub_p.shape)
			save_path = os.path.join(save_dir, '{}.txt'.format(prefix + '_' + sub_name))
			np.savetxt(save_path, sub_p)
	else:
		r""" transpose weight of dense layer. """
		if p_name == 'weight':
			parameter = np.swapaxes(parameter, 0, 1)
		save_path = os.path.join(save_dir, '{}.txt'.format(prefix + '_' + p_name))
		np.savetxt(save_path, parameter)


def save_model_c_params(model_path: str, save_dir: str):
	r"""
	Save the models param to the directory `save_dir`.
	"""
	model = torch.load(model_path, map_location='cpu')
	for name, p in model.named_parameters():
		save_single_c_param(name, p, save_dir)


# save_model_c_params(model_path=r'D:\python_works\audio_classify\model\config1-model.pkl',
# 					save_dir=r'D:\python_works\audio_classify\model\config2')
