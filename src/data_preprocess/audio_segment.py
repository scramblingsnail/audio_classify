import torch
import torchaudio
import os
import numpy as np


def get_speakers(segment_data_dir):
    speakers_file = os.path.join(segment_data_dir, 'speakers.txt')

    if not os.path.exists(speakers_file):
        return []

    speakers = np.loadtxt(speakers_file).astype(int)
    return list(speakers)


def save_speakers(segment_data_dir, exist_speakers):
    speakers_file = os.path.join(segment_data_dir, 'speakers.txt')
    np.savetxt(speakers_file, list(exist_speakers), fmt='%d')


def mk_label_dir(segment_data_dir, label: int):
    dirs = os.listdir(segment_data_dir)
    label_name = 'label-{:d}'.format(label)
    label_dir = os.path.join(segment_data_dir, label_name)
    if label_name not in dirs:
        os.mkdir(label_dir)
    return label_dir


def save_audio_segments(raw_data_dir):
    r""" speaker-label-voice_idx """
    r""" 13-6, 13-7, 19-9, 21-9, 22-6 in board not saved  """
    print('Segmenting audio waves ...')
    segment_dir = os.path.join(raw_data_dir, 'segment')

    root = __file__
    for i in range(3):
        root = os.path.dirname(root)
    segment_data_dir = r'{}/data/voices'.format(root)
    if not os.path.exists(segment_data_dir):
        os.mkdir(segment_data_dir)

    exist_speakers = set(get_speakers(segment_data_dir))
    start_speaker_idx = 0 if len(exist_speakers) == 0 else max(exist_speakers) + 1
    speaker_voices_count = {start_speaker_idx: 0}

    audio_files = os.listdir(raw_data_dir)
    # speaker-label-seg_idx.wav
    for audio_f_name in audio_files:
        if audio_f_name[-3:] != 'wav':
            continue

        audio_path = os.path.join(raw_data_dir, audio_f_name)
        meta_data = torchaudio.info(audio_path)
        sr, sample_bits = meta_data.sample_rate, meta_data.bits_per_sample
        wav, sr = torchaudio.load(audio_path)

        audio_prefix = audio_f_name.split('.')[0]
        speaker_index, label = [int(item) for item in audio_prefix.split('-')[:2]]
        speaker_index += start_speaker_idx
        exist_speakers.add(speaker_index)
        label_dir = mk_label_dir(segment_data_dir, label)

        segment_path = os.path.join(segment_dir, '{}.txt'.format(audio_prefix))
        segment_indices = np.loadtxt(segment_path, delimiter=',').astype(int)
        if len(segment_indices.shape) < 2 and segment_indices.shape[-1] != 0:
            segment_indices = segment_indices[np.newaxis, :]

        speaker_label = '{:d}-{:d}'.format(speaker_index, label)
        for idx in range(segment_indices.shape[0]):
            start, end = tuple(segment_indices[idx])
            wav_segment = wav[:, start:end]
            if speaker_index not in speaker_voices_count:
                speaker_voices_count[speaker_index] = 0
            else:
                speaker_voices_count[speaker_index] += 1

            voice_data_name = speaker_label + '-{:d}.wav'.format(speaker_voices_count[speaker_index])
            voice_data_path = os.path.join(label_dir, voice_data_name)
            # print('from {} to {}'.format(start, end))
            # print('{} saved.'.format(voice_data_name))
            torchaudio.save(voice_data_path, wav_segment, sr, bits_per_sample=sample_bits)

    save_speakers(segment_data_dir, exist_speakers)


if __name__ == '__main__':
    save_audio_segments(raw_data_dir=r'D:/python_works/audio_classify/data/board')
    save_audio_segments(raw_data_dir=r'D:/python_works/audio_classify/data/pc_single_ch')
