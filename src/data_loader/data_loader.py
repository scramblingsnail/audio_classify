import torch
import numpy as np
import h5py

from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):
	def __init__(self, data, labels, lens):
		super().__init__()
		assert data.size()[0] == labels.size()[0] == lens.size()[0]
		self.data = data
		self.labels = labels
		self.lens = lens
		self.data_num = data.size()[0]

	def __getitem__(self, idx):
		return self.data[idx], self.labels[idx], self.lens[idx]

	def __len__(self):
		return self.data_num


def label_mapping(labels: np.ndarray, mapping: dict = None):
	r"""
	map the raw labels to labels for classification, each label contains two value.
	the first value is used for binary classification, while the second value is used for multiclass classification.

	Args:
		labels (np.ndarray): raw labels.
			candidate values: [1, 3, 6, 7, 8, 9]

		mapping (dict): map the raw label to the labels for classification.
			the mapping is as follows:
				```python
				{1: (0, 0), 3: (0, 0), 6: (1, 0), 7: (0, 1), 8: (0, 2), 9: (0, 3)}
				```
	"""
	if mapping is None:
		mapping = {1: (0, 0), 3: (0, 0), 6: (1, 0), 7: (0, 1), 8: (0, 2), 9: (0, 3)}

	new_labels = np.zeros((labels.shape[0], 2)).astype(int)
	for raw_label in mapping.keys():
		new_labels[labels==raw_label] = np.broadcast_to(np.array(mapping[raw_label]).astype(int),
														((labels==raw_label).sum(), 2))
	return new_labels


def read_h5_data(data_path: str, device: torch.device):
	r"""
	read data from h5 file, and transform it to match the model.

	Args:
		data_path (str): data path.
		device (torch.device): the map_location of data and model.

	Returns:
		features (torch.Tensor): the features with size (N, L, INPUT_SIZE)
		labels (torch.tensor): the transformed labels with size (N, 2)
		lens (torch.tensor): the lengths of sequences with size (N,)
	"""
	with h5py.File(data_path, 'r') as data_f:
		features, labels, lens = data_f['data'][:], data_f['labels'][:], data_f['lens'][:]

	features = np.swapaxes(features, 1, 2)
	labels = label_mapping(labels)

	features = torch.from_numpy(features).float()
	labels = torch.from_numpy(labels).long()
	lens = torch.from_numpy(lens).long()
	features = features.to(device)
	labels = labels.to(device)
	lens = lens.to(device)
	return features, labels, lens


def get_loaders(train_data_path, test_data_path, batch_size, device):
	r""" get train data loader and test data loader """
	if isinstance(device, str):
		device = torch.device(device)
	train_data, train_labels, train_lens = read_h5_data(train_data_path, device)
	test_data, test_labels, test_lens = read_h5_data(test_data_path, device)

	train_dataset = AudioDataset(train_data, train_labels, train_lens)
	test_dataset = AudioDataset(test_data, test_labels, test_lens)

	train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
	return train_dataloader, test_dataloader
