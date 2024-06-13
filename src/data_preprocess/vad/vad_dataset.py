import torch
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
import sys

sys.path.append(str(Path(__file__).parent))
from filter.filter_banks import Filter
from util import read_wav, sample_rate_to_8K

def read_label_txt(label_path):
    labels = []
    with open(label_path, "r") as f:
        label = f.readlines()
        label = [list(map(int, i.split(","))) for i in label]
        labels.extend(label)
    # print(labels)
    return np.array(labels)

def get_train_data(
    FS=8000, FRAME_T=0.032, FRAME_STEP=0.016, VOICE_FRAME=4, UNVOICE_FRAME=8
):
    
    train_voice_segments = []
    train_voice_label = []

    data_dir = str(Path(__file__).parent) + "/data"
    label_dir = str(Path(__file__).parent) + "/label"

    filter = Filter()
    for file_dir in Path(data_dir).iterdir():
        if file_dir.name[-3:] != "wav":
            continue
        
        label_file_name = label_dir +'/'+ file_dir.name[:-3] + "txt"
        labels = read_label_txt(label_file_name)
        print(labels.shape)
        signal, signal_len, sample_rate = read_wav(str(file_dir))


        signal, signal_len = sample_rate_to_8K(signal, sample_rate)

        for i in range(0, signal_len, int(FRAME_STEP * FS)):
            if i + FS * FRAME_T - 1 > signal_len:
                break

            tmp_signal = signal[i : int(i + FS * FRAME_T)]
            tmp_signal = filter.energy_filter(tmp_signal)

            train_voice_segments.append(tmp_signal)
            
            label_num = 0
            right_index = i + FS * FRAME_T - 1
            for label in labels:
                if i >= label[0] and right_index <= label[1]:
                    label_num = 1
                # elif i >= label[0] and right_index >= label[1]:
                #     label_num = [1,0]
                # elif i <= label[0] and right_index <= label[1]:
                #     label_num = [0,1]

            train_voice_label.append(label_num)
    
    train_voice_segments = np.array(train_voice_segments)
    print(train_voice_segments.shape)
    train_voice_label = np.array(train_voice_label)
    # print(train_voice_label.shape)
    return train_voice_segments, train_voice_label
    
class VAD_Dataset(Dataset):
    def __init__(self):
        self.data, self.label = get_train_data()
        self.data = self.data.reshape(-1, 1, 129, 1)
        self.data = torch.from_numpy(self.data).float()
        self.label = torch.from_numpy(self.label).long()

        self.filter = Filter()
        print('Data shape:', self.data.shape)
        print('Label shape:', self.label.shape)
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx, :], self.label[idx]

