# Audio Classification using LSTM

## 基于LSTM的语音片段分类
本部分在整个工作流程中的位置为如下工作流程中的加粗部分：

### 工作流程简述
我们设计了一个语音处理Pipeline, 将前级的VAD、特征提取、后级的分类计算作为一个整体。譬如，在我们的方案中，特征并不是提取完成后再进行处理，而是
利用LSTM作为分类器，再逐帧进行VAD判断与特征提取的同时进行LSTM的计算，因此能极大提高计算效率。工作流程简述如下：

- VAD判断当前语音帧是否存在语音活动，如果存在，则该帧为有效语音帧，将当前语音帧送入下一级处理。
- 对有效语音帧进行`MFCC`特征提取。
- **将该语音帧对应的`MFCC`特征向量作为`LSTM cell`单个时间步的输入。**
- **`LSTM cell` 持续进行迭代计算，直到没有有效语音帧。最后将`LSTM cell`的`hidden states`
输入最后的`Fully Connected Layer` 进行分类，判断该语音片段的种类。**


### 数据处理

1. LSTM 的训练与测试数据来自于 `pc` 与 `board` 目录下的语音文件。其中`pc` 中的语音文件
被转为了`sample rate`为`8000`, 单声道的语音文件。如果需要进行测试，请将`pc`与`board`数据放在[data](../data)目录下。
2. 利用VAD截取出语音文件中的有效语音片段作为训练与测试数据。在截取过程中，我们对分割点进行了随机的
扰动，以模拟实际情况下VAD的非理想性。
3. 从这些语音片段中提取出 `MFCC` 特征，并以 8:2的比例，对每个`label`的说话人进行训练集-测试集分割。
4. 计算训练集与测试集各自在特征维度上的均值与标准差，将训练与测试数据进行标准化。

[mfcc](./src/feature_extract/audio_features.py) or [mel-spectrogram](./src/feature_extract/audio_features.py). 
这些提取的特征将保存在目录[data](./data)的子目录 `mfcc` or `mel-spectrogram` 

### 网络训练
采用的网络结构如下：[模型定义](../src/net/lstm.py)

```python
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
```

其中输出层的前两个输出值用于进行唤醒词 `Hi 芯原` 的二分类，剩余输出值用于进行 `命令` 的多分类。
在这个例子中，共有`4`中命令，其中有一类代表`无效命令`。损失函数采用`CrossEntropyLoss`。

### 训练结果:
训练完成的LSTM分类器，在测试集上取得了 二分类准确率 `99%`，多分类准确率 `95%` 的测试结果。

```text
Epoch 0 - loss: 1.5698 - lr: 0.0040
Epoch 1 - loss: 1.3784 - lr: 0.0040
Epoch 2 - loss: 1.1866 - lr: 0.0040
Epoch 3 - loss: 1.1445 - lr: 0.0040
Epoch 4 - loss: 1.1126 - lr: 0.0040
Epoch 5 - loss: 1.1130 - lr: 0.0040
Epoch 6 - loss: 1.1007 - lr: 0.0040
Epoch 7 - loss: 1.0854 - lr: 0.0040
Epoch 8 - loss: 1.0854 - lr: 0.0040
Epoch 9 - loss: 1.0766 - lr: 0.0040
- binary test acc: 0.96, - multi test acc: 0.93
Epoch 10 - loss: 1.0894 - lr: 0.0040
Epoch 11 - loss: 1.0966 - lr: 0.0040
Epoch 12 - loss: 1.0898 - lr: 0.0040
Epoch 13 - loss: 1.0906 - lr: 0.0020
Epoch 14 - loss: 1.0855 - lr: 0.0020
Epoch 15 - loss: 1.0743 - lr: 0.0020
Epoch 16 - loss: 1.0710 - lr: 0.0020
Epoch 17 - loss: 1.0723 - lr: 0.0020
Epoch 18 - loss: 1.0705 - lr: 0.0020
Epoch 19 - loss: 1.0693 - lr: 0.0020
- binary test acc: 0.99, - multi test acc: 0.94
Epoch 20 - loss: 1.0696 - lr: 0.0020
Epoch 21 - loss: 1.0681 - lr: 0.0020
Epoch 22 - loss: 1.0668 - lr: 0.0020
Epoch 23 - loss: 1.0677 - lr: 0.0020
Epoch 24 - loss: 1.0741 - lr: 0.0020
Epoch 25 - loss: 1.0687 - lr: 0.0020
Epoch 26 - loss: 1.0674 - lr: 0.0010
Epoch 27 - loss: 1.0668 - lr: 0.0010
Epoch 28 - loss: 1.0667 - lr: 0.0010
Epoch 29 - loss: 1.0667 - lr: 0.0010
- binary test acc: 0.99, - multi test acc: 0.95
Epoch 30 - loss: 1.0667 - lr: 0.0005
Epoch 31 - loss: 1.0668 - lr: 0.0005
Epoch 32 - loss: 1.0666 - lr: 0.0005
Epoch 33 - loss: 1.0669 - lr: 0.0005
Epoch 34 - loss: 1.0668 - lr: 0.0005
Epoch 35 - loss: 1.0666 - lr: 0.0005
Epoch 36 - loss: 1.0665 - lr: 0.0003
Epoch 37 - loss: 1.0665 - lr: 0.0003
Epoch 38 - loss: 1.0665 - lr: 0.0003
Epoch 39 - loss: 1.0665 - lr: 0.0003
- binary test acc: 0.99, - multi test acc: 0.95
Epoch 40 - loss: 1.0666 - lr: 0.0003
Epoch 41 - loss: 1.0665 - lr: 0.0001
Epoch 42 - loss: 1.0666 - lr: 0.0001
Epoch 43 - loss: 1.0664 - lr: 0.0001
Epoch 44 - loss: 1.0666 - lr: 0.0001
Epoch 45 - loss: 1.0664 - lr: 0.0001
Epoch 46 - loss: 1.0664 - lr: 0.0001
Epoch 47 - loss: 1.0664 - lr: 0.0001
Epoch 48 - loss: 1.0664 - lr: 0.0001
Epoch 49 - loss: 1.0664 - lr: 0.0001
- binary test acc: 0.99, - multi test acc: 0.95
```

### 快速开始
我们提供了便捷的训练测试入口。

- 在[这个目录下](./src/cfg)创建你的配置文件(`XXX.yaml`)，你可以参考 [config1](./src/cfg/config1.yaml) 作为一个例子。
- 转到[运行脚本run](./run.py), 将对应的参数`config_flag`修改为你配置文件的名字`XXX`。
- run