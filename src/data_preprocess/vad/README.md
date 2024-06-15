# 简述
此目录为基于CNN模型的VAD算法的python仿真代码。

# 文件说明
model.py 
- 定义了用于VAD预测的CNN网络结构

vad_dataset.py 
- 定义了用于VAD检测的数据集，其根据label和wav数据生成用于CNN训练的数据集

VAD.py 
- 定义了VAD预测的流程

test_model.py 
- 包含了数据读取，流式处理和预测的功能，并且测试过程中生成predict txt文件，用于进行evaluate分数

util.py
- 提供了相应的辅助函数，包括读取wav文件，降采样和画图等等

model/
- 存放已训练的CNN模型，pth格式
- `model_microphone.pth`为参考代码中的模型
- `model.pth`为本项目训练得到的模型

data/
- 用于测试该代码的原始audio数据。

label/
- 存储audio数据对应的label

filter/
- 存储了滤波器组的代码
- CNN可以配合滤波器组，对语音的能量谱进行预测来实现更精确的VAD预测，考虑到单片机性能/数据集过小，当前版本暂未加入。

predict/
- 模型的预测结果，由`main.py`生成，并可以被`evaluate.py`进行评估
- 该文件并未被上传到github，可以执行代码进行生成

# 数据集生成方法

语音信号会被分为大量的语音片段，其被作为数据输入到神经网络中，`vad_dataset`负责处理该文件。

Label根据语音片段和Label文件进行生成，如果语音片段完全位于标记的语音信号片段中，则Label为1，否则为0。例如，`[100, 1000]`被标记为语音信号，那么对于片段`[0:240]`，其label为0，对于片段`[240, 480]`其label为1.

## 训练数据

训练集数据主要分为两个部分

1. 作业中的data数据
2. 使用预训练VAD对SDK中的board、pc数据进行label标记，然后用再被用于训练VAD

# 神经网络

**CNN网络**，使用大小为2X1的卷积核进行滑动卷积，slide设置为2，并在CNN后接入了一个全连接层，具体参见`model.py`。
- 使用交叉熵函数计算Loss值，以及使用了Adam算法对网络进行优化。

## 网络结构

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

# 分数评价 —— evaluate_score.py

运行该文件，并更改相应参数，即可得到评估结果。默认评估对data_1的预测结果。
- 参数

```python
data_num = 1
WAV = dir_path + f'/data/data_{data_num}.wav'
LABEL_INPUT = dir_path + f'/label/data_{data_num}.txt'
PREDICT_INPUT = dir_path + f'/predict/data_{data_num}.txt'
```