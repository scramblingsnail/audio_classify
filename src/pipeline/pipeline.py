import torch
import os

from torch.optim import Adam

from .train import train
from .test import test
from .utils import prepare_data, set_device, set_seed
from ..cfg import load_config
from ..net import AudioClassifier
from ..feature_extract import get_all_features
from ..data_loader import get_loaders


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
	model_path = os.path.join(root, r'model/{}-model.pkl'.format(config_flag))

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

	max_bin_acc, max_multi_acc = 0, 0
	for epoch in range(config['epochs']):
		train_loss = train(model=model, train_loader=train_dataloader, optimizer=optimizer, loss_func=loss_func)
		lr_reduction.step(train_loss)
		print('Epoch {} - loss: {:.4f} - lr: {:.4f}'.format(epoch, train_loss,optimizer.param_groups[0]['lr']))
		if (epoch + 1) % config['evaluate_interval'] == 0:
			bin_acc, multi_acc = test(model=model, test_loader=test_dataloader)
			if bin_acc > max_bin_acc and multi_acc > max_multi_acc:
				max_bin_acc, max_multi_acc = bin_acc, multi_acc
				torch.save(model, model_path)
