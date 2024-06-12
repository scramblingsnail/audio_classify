import yaml
import os


def load_config(config_flag: str = None):
	if config_flag is None:
		config_flag = 'config1'

	cfg_dir = os.path.dirname(__file__)
	config_path = os.path.join(cfg_dir, '{}.yaml'.format(config_flag))

	if not os.path.exists(config_path):
		raise ValueError('config file do not exist: {}'.format(config_path))

	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)

	config['config_flag'] = config_flag
	if config['feature_func_name'] == 'mfcc':
		config['input_size'] = config['n_mfcc']
	else:
		config['input_size'] = config['n_mels']
	return config
