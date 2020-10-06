import librosa
import torchaudio
import numpy as np
from torch import Tensor
from scipy.io.wavfile import read
from tacotron2.utils import logger


class AudioParser(object):
    def __init__(self, *args, **kwargs):
        super(AudioParser, self).__init__()

    def load_audio(self, audio_path):
        try:
            sample_rate, signal = read(audio_path)

            return signal / 32767  # normalize audio

        except ValueError:
            logger.debug('ValueError in {0}'.format(audio_path))
            return None
        except RuntimeError:
            logger.debug('RuntimeError in {0}'.format(audio_path))
            return None
        except IOError:
            logger.debug('IOError in {0}'.format(audio_path))
            return None

    def parse_audio(self, *args, **kwargs):
        raise NotImplementedError


class MelSpectrogramParser(AudioParser):
    def __init__(
            self,
            feature_extract_by: str = 'librosa',     # which library to use for feature extraction
            sample_rate: int = 22050,                # sample rate of audio signal.
            num_mel_bins: int = 80,                  # Number of mfc coefficients to retain.
            frame_length_ms: float = 50.0,           # frame length for spectrogram
            frame_shift_ms: float = 12.5,            # Length of hop between STFT windows.
    ):
        super(MelSpectrogramParser, self).__init__()
        self.feature_extract_by = feature_extract_by
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.n_fft = int(round(sample_rate * 0.001 * frame_length_ms))
        self.hop_length = int(round(sample_rate * 0.001 * frame_shift_ms))

        if feature_extract_by == 'torchaudio':
            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
            self.transforms = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, win_length=frame_length_ms,
                hop_length=self.hop_length, n_fft=self.n_fft,
                n_mels=num_mel_bins
            )

    def parse_audio(self, audio_path):
        signal = self.load_audio(audio_path)

        if self.feature_extract_by == 'torchaudio':
            mel_spectrogram = self.transforms(Tensor(signal))
            mel_spectrogram = self.amplitude_to_db(mel_spectrogram)

        elif self.feature_extract_by == 'librosa':
            mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=self.sample_rate, n_mels=self.num_mel_bins,
                                                             n_fft=self.n_fft, hop_length=self.hop_length)
            mel_spectrogram = self.amplitude_to_db(mel_spectrogram, ref=np.max)

        else:
            raise ValueError("Unsupported library : {0}".format(self.feature_extract_by))

        return mel_spectrogram