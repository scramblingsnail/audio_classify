import torch

from torch.utils.data import DataLoader
from torch.optim import Optimizer

from ..net import AudioClassifier


def train(model, train_loader: DataLoader, optimizer: Optimizer, loss_func):
	model.train()
	avg_loss = 0
	for batch_data, batch_labels, batch_lens in train_loader:
		batch_bin_p, batch_multi_p = model(batch_data, batch_lens)
		bin_loss = loss_func(batch_bin_p, batch_labels[:, 0])
		multi_loss = loss_func(batch_multi_p, batch_labels[:, 1])
		loss = bin_loss + multi_loss
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		avg_loss += loss.detach().cpu().item()

	avg_loss /= len(train_loader)
	return avg_loss





