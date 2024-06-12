import torch
import torch.nn as nn


class AudioClassifier(nn.Module):
	def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
		super().__init__()
		self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
							bidirectional=False)
		self.fc = nn.Linear(hidden_size, output_size)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x: torch.Tensor, lens: torch.Tensor):
		r"""

		Args:
			x (torch.Tensor): input features with size (N, L, INPUT_SIZE)
			lens (torch.Tensor): sequence lengths with size (N, )

		Returns:
			bin_p (torch.Tensor): the possibility vector for binary classification.
			multi_p (torch.Tensor): the possibility vector for multi-class classification.
		"""
		# N, L, H_{out}
		x, _ = self.lstm(x)
		# N, 1, H_{out}
		x = torch.gather(x, dim=1, index=(lens - 1).unsqueeze(1).unsqueeze(2).expand(x.size()[0], 1, x.size()[2]))
		x = x.squeeze(1)
		x = self.fc(x)
		bin_p = self.softmax(x[:, :2])
		multi_p = self.softmax(x[:, 2:])
		return bin_p, multi_p
