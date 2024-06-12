import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from util import *
from vad import VAD


def cal_voice_segment(pred_class, pred_idx_in_data, raw_data_len):
    """
    Determine the start and end points of the vocal segment based on the prediction results.

    param: pre_calss: (N,1),Predicted calss of human voice. 0: unvoice, 1:voice
    param: pred_idx_in_data: (N,1),Each prediction result corresponds to the coordinates on the original data
    param: raw_data_len: the length of raw data
    return: (M,2), voice segment. (...,0):start point, (...,1):end point
    """

    if len(pred_class) != len(pred_idx_in_data):
        raise Exception("pred_class length must be pred_idx_in_data length!")

    all_voice_segment = np.array([])
    single_voice_segment = []
    diff_value = np.diff(pred_class)

    for i in range(len(diff_value)):
        if diff_value[i] == 1:
            single_voice_segment.append(pred_idx_in_data[i + 1])

        if diff_value[i] == -1:
            if len(single_voice_segment) == 0:
                single_voice_segment.append(0)
            single_voice_segment.append(pred_idx_in_data[i + 1])

        if len(single_voice_segment) == 2:
            if len(all_voice_segment) == 0:
                all_voice_segment = np.array(single_voice_segment).reshape(1, -1)
            else:
                all_voice_segment = np.concatenate(
                    (all_voice_segment, np.array(single_voice_segment).reshape(1, -1)),
                    axis=0,
                )
            single_voice_segment = []

    if len(single_voice_segment) == 1:
        single_voice_segment.append(raw_data_len - 1)
        all_voice_segment = np.concatenate(
            (all_voice_segment, np.array(single_voice_segment).reshape(1, -1)), axis=0
        )

    return all_voice_segment


def vad_forward(data_dir: str, model_path: str):
    vad_model = VAD(model_path=model_path)

    for file_dir in Path(data_dir).iterdir():

        if file_dir.name[-3:] != "wav":
            continue

        signal, signal_len, sample_rate = read_wav(str(file_dir))
        print(file_dir, sample_rate)

        signal, signal_len = sample_rate_to_8K(signal, sample_rate)
        # signal = signal * 10

        total_pred = np.array([])
        total_indices = np.array([])
        for i in range(0, signal_len, int(FRAME_STEP * FS)):
            if i + FS * FRAME_T - 1 > signal_len:
                break

            tmp_signal = signal[i : int(i + FS * FRAME_T)]

            pred = vad_model.process(tmp_signal)

            if total_indices.size == 0:
                total_indices = np.array(i)
            else:
                total_indices = np.concatenate((total_indices, i), axis=None)

            if total_pred.size == 0:
                total_pred = pred.copy()
            else:
                total_pred = np.concatenate((total_pred, pred), axis=None)

        voice_segment = cal_voice_segment(total_pred, total_indices, signal_len)

        dir_path = str(Path(__file__).parent)
        # with open(dir_path + "/predict/" + file_dir.name[:-3] + "txt", "w") as file:
        #     for segment in voice_segment:
        #         file.write(str(segment[0]) + "," + str(segment[1]) + "\n")
        plt.figure(1, figsize=(15, 7))
        plt.clf()
        draw_time_domain_image(
            signal, nframes=signal_len, framerate=sample_rate, line_style="b-"
        )
        # print(voice_segment)
        draw_result(signal, voice_segment)
        plt.legend(["signal", "voice segment"])

        plt.grid()
        plt.show()


if __name__ == "__main__":
    FS = 8000
    FRAME_T = 0.03
    FRAME_STEP = 0.015
    VOICE_FRAME = 4
    UNVOICE_FRAME = 8
    
    parent_path = Path(__file__).parent.parent.parent
    print(parent_path)
    model_path = str(parent_path) + "/model/model.pth"
    data_dir = str(parent_path) + "/data/pc"
    print(model_path)
    vad_forward(data_dir=data_dir, model_path=model_path)
