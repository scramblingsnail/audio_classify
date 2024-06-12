import torch
import os


def test(model, test_loader):
	model.eval()

	bin_correct = 0
	multi_correct = 0
	for batch_data, batch_labels, batch_lens in test_loader:
		batch_bin_p, batch_multi_p = model(batch_data, batch_lens)

		batch_bin_predict = torch.argmax(batch_bin_p, dim=-1)
		batch_multi_predict = torch.argmax(batch_multi_p, dim=-1)

		batch_bin_correct = (batch_bin_predict == batch_labels[:, 0]).sum()
		batch_multi_correct = (batch_multi_predict == batch_labels[:, 1]).sum()
		bin_correct += batch_bin_correct.item()
		multi_correct += batch_multi_correct.item()

	bin_correct /= len(test_loader.dataset)
	multi_correct /= len(test_loader.dataset)
	print('- binary test acc: {:.2f}, - multi test acc: {:.2f}'.format(bin_correct, multi_correct))
	return bin_correct, multi_correct
