from math import floor
import torch
import torchaudio
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

class FilterBanks:
    def __init__(self) -> None:
        # self.worksapce_dir = Path(__file__).parent.parent.parent.parent
        # print(f'Filter banks workspace directory: {self.worksapce_dir}')
        # self.pc_ch_dir = self.worksapce_dir / 'data' / 'pc_single_ch'
        # print(f'PC single channel directory: {self.pc_ch_dir}')
        # self.board_dir = self.worksapce_dir / 'data' / 'board'
        pass

    def mel_filter_banks(self, n_fft, n_mels, sample_rate):
        """
        Create a mel filter banks
        """
        mel_filter_banks = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            n_stft=n_fft,
            f_min=0.0,
            f_max=sample_rate / 2,
        )
        return mel_filter_banks

    def energy_filter_banks(self, n_fft=256, sample_rate=8000, power=2.0):
        """
        Create an energy filter banks
        """
        energy_filter_banks = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=floor(n_fft * 0.5),
            win_length=n_fft,
            # window_fn=torch.hann_window,
            center=False,
            pad_mode="constant",
            power=power,
            normalized=True,
        )
        return energy_filter_banks


class Filter:
    def __init__(self) -> None:
        filter_banks = FilterBanks()
        self.energy_filter_banks = filter_banks.energy_filter_banks()
        self.mel_filter_banks = filter_banks.mel_filter_banks(
            n_fft=2048, n_mels=128, sample_rate=8000
        )

    def energy_filter(self, waveform):
        waveform = torch.as_tensor(waveform).float()
        # print(f'Waveform shape: {waveform.dtype}')
        return Filter.power_to_db(self.energy_filter_banks(waveform)).squeeze(0)

    @staticmethod
    def power_to_db(power, ref=1.0, amin=1e-10, top_db=80.0):
        """
        Convert a power spectrogram (amplitude squared) to decibel (dB) units
        This function is based on the implementation from librosa
        """
        power = torch.as_tensor(power)
        ref = torch.as_tensor(ref)
        amin = torch.as_tensor(amin)
        top_db = torch.as_tensor(top_db)

        # scale the input power
        log_spec = 10.0 * torch.log10(torch.clip(power, amin))

        # scale the output
        log_spec -= 10.0 * torch.log10(torch.maximum(amin, ref))

        if top_db is not None:
            if top_db < 0:
                raise ValueError("top_db must be non-negative")
            log_spec = torch.maximum(log_spec, log_spec.max() - top_db)

        return log_spec


if __name__ == "__main__":
    # filter_banks = FilterBanks()
    worksapce_dir = Path(__file__).parent.parent.parent.parent.parent
    pc_ch_dir = worksapce_dir / "data" / "pc_single_ch"
    board_dir = worksapce_dir / "data" / "board"

    filter = Filter()

    wave_points, sample_rate = torchaudio.load(board_dir / "0-9.wav")


    plt.figure(1)
    plt.imshow(filter.energy_filter(wave_points))
    plt.show()
