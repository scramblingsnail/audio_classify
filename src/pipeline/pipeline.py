import torch
import os
import numpy as np

from torch.optim import Adam

from .train import train
from .test import test
from .utils import prepare_data, set_device, set_seed
from ..cfg import load_config
from ..net import AudioClassifier, save_model_c_params
from ..feature_extract import get_all_features
from ..data_loader import get_loaders
from ..feature_extract import mfcc_features


def classify_pipeline(config_flag: str = None):
	r"""
	A pipeline including:

	- feature extraction
	"""
	config = load_config(config_flag)
	prepare_data()

	root = __file__
	for i in range(3):
		root = os.path.dirname(root)
	config['root'] = root

	train_data_path, test_data_path = get_all_features(config=config)
	model_dir = os.path.join(root, r'model')
	print(root)
	print(model_dir)
	if not os.path.exists(model_dir):
		os.mkdir(model_dir)

	config_model_dir = os.path.join(model_dir, config_flag)
	if not os.path.exists(config_model_dir):
		os.mkdir(config_model_dir)

	model_path = os.path.join(config_model_dir, r'{}-model.pkl'.format(config_flag))

	device = set_device(config['device'])
	set_seed(config['random_seed'])
	train_dataloader, test_dataloader = get_loaders(batch_size=config['batch_size'], train_data_path=train_data_path,
													test_data_path=test_data_path, device=device)

	model = AudioClassifier(input_size=config['input_size'], hidden_size=config['hidden_size'],
							num_layers=config['num_layers'], output_size=config['output_size'])
	model.to(device)
	optimizer = Adam(params=model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
	loss_func = torch.nn.CrossEntropyLoss()

	lr_reduction = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
															  factor=config['lr_reduction_factor'],
															  patience=config['lr_patience'], min_lr=config['min_lr'],
															  threshold=config['metric_threshold'],
															  threshold_mode='rel', cooldown=config['cool_down'])
	# train loop
	max_bin_acc, max_multi_acc = 0, 0
	for epoch in range(config['epochs']):
		train_loss = train(model=model, train_loader=train_dataloader, optimizer=optimizer, loss_func=loss_func)
		lr_reduction.step(train_loss)
		print('Epoch {} - loss: {:.4f} - lr: {:.4f}'.format(epoch, train_loss, optimizer.param_groups[0]['lr']))
		if (epoch + 1) % config['evaluate_interval'] == 0:
			bin_acc, multi_acc = test(model=model, test_loader=test_dataloader)
			if bin_acc > max_bin_acc and multi_acc > max_multi_acc:
				max_bin_acc, max_multi_acc = bin_acc, multi_acc
				torch.save(model, model_path)

	# save parameters for c:
	save_model_c_params(model_path=model_path, save_dir=config_model_dir)


def test_pipeline(config_flag):
	config = load_config(config_flag)
	root = __file__
	for i in range(3):
		root = os.path.dirname(root)
	config['root'] = root
	train_data_path, test_data_path = get_all_features(config=config)
	device = set_device(config['device'])

	_, test_dataloader = get_loaders(batch_size=config['batch_size'], train_data_path=train_data_path,
									 test_data_path=test_data_path, device=device)
	model_dir = os.path.join(root, r'model')
	config_model_dir = os.path.join(model_dir, config_flag)
	model_path = os.path.join(config_model_dir, r'{}-model.pkl'.format(config_flag))
	model = torch.load(model_path, map_location=device)
	test(model=model, test_loader=test_dataloader)


def debug(config_flag):
	config = load_config(config_flag)
	prepare_data()

	root = __file__
	for i in range(3):
		root = os.path.dirname(root)
	config['root'] = root

	device = set_device(config['device'])
	model_dir = os.path.join(root, r'model')
	config_model_dir = os.path.join(model_dir, config_flag)
	model_path = os.path.join(config_model_dir, r'{}-model.pkl'.format(config_flag))
	model = torch.load(model_path, map_location=device)

	for wav_idx in range(7):
		wav_path = os.path.join(root, 'data/qemu/{:d}.wav'.format(wav_idx))
		mean_path = os.path.join(root, 'data/mfcc/{}_all_mean.txt'.format(config_flag))
		std_path = os.path.join(root, 'data/mfcc/{}_all_std.txt'.format(config_flag))

		mean = np.loadtxt(mean_path)
		std = np.loadtxt(std_path)

		mean = torch.as_tensor(mean, dtype=torch.float, device=device).unsqueeze(1)
		std = torch.as_tensor(std, dtype=torch.float, device=device).unsqueeze(1)
		# print(mean.size(), mean)
		# print(std.size(), std)

		mfcc = mfcc_features(wav_path=wav_path, config=config)
		# print(mfcc.size())
		mfcc = mfcc.squeeze(0).to(device)
		mfcc = (mfcc - mean) / std
		mfcc = torch.transpose(mfcc, 0, 1)
		lens = torch.tensor([mfcc.size()[1]], dtype=torch.long, device=device)
		# print(lens)

		inputs = mfcc.unsqueeze(0)
		output = model.forward(inputs, lens)
		binary, multi =  output[0].detach(), output[1].detach()
		print('binary: ', binary, 'multi: ', multi)
		print('\t>>> label: binary: ', torch.argmax(binary).cpu().numpy(), 'multi: ', torch.argmax(multi).cpu().numpy())



