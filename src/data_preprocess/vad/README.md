# 简述
	此目录为基于CNN模型的VAD算法的python仿真代码。

# 文件说明
model.py 
- 定义了用于VAD预测的CNN网络结构

vad_dataset.py 
- 定义了用于VAD检测的数据集，其根据label和wav数据生成用于CNN训练的数据集

VAD.py 
- 定义了VAD预测的流程

main.py 
- 算法测试的主函数，其中包含了数据读取，流式处理和预测的功能，并且测试过程中生成predict txt文件，用于进行evaluate

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

predict/
- 模型的预测结果，由`main.py`生成，并可以被`evaluate.py`进行评估

# 数据集生成方法

语音信号会被分为大量的语音片段（240个点），其被作为数据输入到神经网络中，大小为`[N, 1, 1, 240]`。

Label根据语音片段和Label文件进行生成，如果语音片段完全位于标记的语音信号片段中，则Label为1，否则为0。例如，`[100, 1000]`被标记为语音信号，那么对于片段`[0:240]`，其label为0，对于片段`[240, 480]`其label为1.

# 神经网络

主要使用了1X2的卷积核进行滑动卷积，slide设置为2，参见`model.py`。
使用了交叉熵函数计算Loss值，以及使用了Adam算法对网络进行优化。

# evaluate.py

运行该文件，并更改相应参数，即可得到评估结果。默认评估对data_1的预测结果。

```python
WAV = dir_path + '/data/data_1.wav'
LABEL_INPUT = dir_path + '/label/data_1.txt'
PREDICT_INPUT = dir_path + '/predict/data_1.txt'
```