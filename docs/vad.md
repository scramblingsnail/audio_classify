# VAD
使用CNN + 全连接层的方式，判断是否有语音信号

## 训练数据

训练集数据主要分为两个部分

1. 作业中的data数据
2. 使用预训练VAD对SDK中的board、pc数据进行label标记，然后用再被用于训练VAD


数据的输入为语音信号片段，输出为二分类的Label，即**有语音活动**和**没有语音活动**。具体实现方式参加`root/src/data_preprocess/vad`目录下的[文档](../src/data_preprocess/vad/README.md)。

## 神经网络结构

**CNN网络**，使用大小为2X1的卷积核进行滑动卷积，slide设置为2，并在CNN后接入了一个全连接层，具体参见`model.py`。

- 使用交叉熵函数计算Loss值，以及使用了Adam算法对网络进行优化。
- 使用softmax函数

```python
class CNN(nn.Module):
    def __init__(
        self,
        input_channel=1,
        n_channel=1,
        kernel_size=2,
        stride=2,
        dilation=1,
        padding="valid",
    ) -> None:
        super(CNN, self).__init__()

        self.fc_size = 128
        model = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=n_channel,
                kernel_size=(kernel_size, 1),
                stride=stride,
                dilation=dilation,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=n_channel),
            nn.Flatten(),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=self.fc_size, out_features=2),
            nn.Softmax(dim=1)
        )
        self.model = model
```
