import torch
import torchaudio
import os
import numpy as np

from torchaudio.functional import resample


def pc_preprocess(pc_data_dir):
    data_dir = os.path.dirname(pc_data_dir)
    new_pc_data_dir = os.path.join(data_dir, 'pc_single_ch')

    if not os.path.exists(new_pc_data_dir):
        os.mkdir(new_pc_data_dir)

    target_sr = 8000
    sample_bits = 16
    files = os.listdir(pc_data_dir)

    for f_name in files:
        f_path = os.path.join(pc_data_dir, f_name)
        wav, sr = torchaudio.load(f_path)
        new_wav = resample(wav, orig_freq=sr, new_freq=target_sr)
        for ch_idx in range(new_wav.size()[0]):
            ch_f_name = f_name.split('.')[0] + '-ch{:d}'.format(ch_idx) + '.wav'
            ch_f_path = os.path.join(new_pc_data_dir, ch_f_name)
            torchaudio.save(ch_f_path, new_wav[ch_idx:ch_idx+1], target_sr, bits_per_sample=sample_bits)
            print('CH {:d} of {} has been '
                  'resampled as {} with sample rate {:d}'.format(ch_idx, f_name, ch_f_name, target_sr))


def board_preprocess(board_data_dir):
    files = os.listdir(board_data_dir)
    print(files)
    target_sr = 8000
    sample_bits = 16
    data_bytes_num = sample_bits // 8
    amplitude = 2 ** (sample_bits - 1) - 1
    for f_name in files:
        if f_name[-3:] != 'bin':
            continue

        f_path = os.path.join(board_data_dir, f_name)
        size = os.path.getsize(f_path)
        assert size % data_bytes_num == 0
        wav_data = torch.zeros((1, size // data_bytes_num))
        with open(f_path, 'rb') as f:
            for i in range(size):
                if i % data_bytes_num == 0:
                    data = f.read(data_bytes_num)
                    int_data = int.from_bytes(data, byteorder='little', signed=True)
                    wav_data[0, i // data_bytes_num] = int_data / amplitude

        wav_f_name = f_name.split('.')[0] + '.wav'
        wav_f_path = os.path.join(board_data_dir, wav_f_name)
        print('{} has been saved as {}'.format(f_name, wav_f_name))
        torchaudio.save(wav_f_path, wav_data, target_sr, bits_per_sample=sample_bits)


if __name__ == '__main__':
    board_preprocess(r'D:\Datasets\competitions\xinyuan\data\board')
    pc_preprocess(r'D:\python_works\audio_classify\data\pc')