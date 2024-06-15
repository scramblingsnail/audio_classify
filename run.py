from src.pipeline import classify_pipeline, test_pipeline, debug


if __name__ == '__main__':
	r""" modify the 'config_flag' to your own configuration file name. """
	classify_pipeline(config_flag='config2')
	# test_pipeline(config_flag='config1')
	# debug(config_flag='config2')
