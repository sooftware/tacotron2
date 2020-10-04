# -*- coding: utf-8 -*-
# Soohwan Kim @sooftware
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree

import torch
import threading
import librosa
from text import text_to_sequence


class TextMelDataset(object):
    def __init__(
            self,
            dataset_path,
            audio_paths: list,
            transcripts: list,
            sample_rate: int = 220500,
            num_mel_bins: int = 80,
            frame_length_ms: float = 50,
            frame_shift_ms: float = 12.5
    ):
        self.dataset_path = dataset_path
        self.audio_paths = audio_paths
        self.transcripts = transcripts
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.n_fft = int(round(sample_rate * 0.001 * frame_length_ms))
        self.hop_length = int(round(sample_rate * 0.001 * frame_shift_ms))

    def get_text(self, index):
        return torch.IntTensor(text_to_sequence(self.transcripts[index], 'english_cleaner'))

    def get_melspectrogram(self, index):
        signal, sr = librosa.load(self.audio_paths[index], sr=self.sample_rate)
        melspectrogram = librosa.feature.melspectrogram(
            y=signal,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window='hann'
        )
        return melspectrogram

    def get_item(self, index):
        text = self.get_text(index)
        melspectrogram = self.get_melspectrogram(index)

        return text, melspectrogram


class TextMelDataLoader(threading.Thread):
    def __init__(self, dataset: TextMelDataset, queue, batch_size, thread_id):
        super(TextMelDataLoader, self).__init__()

    def run(self):
        raise NotImplementedError


class MultiDataLoader(object):
    """
    Multi Data Loader using Threads.

    Args:
        dataset_list (list): list of MelSpectrogramDataset
        queue (Queue.queue): queue for threading
        batch_size (int): size of batch
        num_workers (int): the number of cpu cores used
    """
    def __init__(self, dataset_list, queue, batch_size, num_workers):
        self.dataset_list = dataset_list
        self.queue = queue
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loader = list()

        for idx in range(self.num_workers):
            self.loader.append(TextMelDataLoader(self.dataset_list[idx], self.queue, self.batch_size, idx))

    def start(self):
        """ Run threads """
        for idx in range(self.num_workers):
            self.loader[idx].start()

    def join(self):
        """ Wait for the other threads """
        for idx in range(self.num_workers):
            self.loader[idx].join()


def split_dataset(args):
    target_dict = load_targets(args.metadata_path, separator='|')


def load_targets(filepath: str, separator: str = '|'):
    target_dict = dict()

    with open(filepath, encoding='utf-8') as f:
        for line in f.readlines():
            audio_path, transcript, _ = line.strip().split(separator)
            target_dict[audio_path] = transcript

    return target_dict
