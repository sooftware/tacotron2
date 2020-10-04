# -*- coding: utf-8 -*-
# Soohwan Kim @sooftware
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree

import torch
import threading
from torch.utils.data import Dataset
from tacotron2.data.audio.parser import MelSpectrogramParser
from tacotron2.data.text import text_to_sequence


class TextMelDataset(Dataset, MelSpectrogramParser):
    def __init__(
            self,
            dataset_path,
            audio_paths: list,
            transcripts: list,
            feature_extract_by: str,
            sample_rate: int = 22050,
            num_mel_bins: int = 80,
            frame_length_ms: float = 50,
            frame_shift_ms: float = 12.5
    ):
        super(TextMelDataset, self).__init__(feature_extract_by, sample_rate, num_mel_bins, frame_length_ms, frame_shift_ms)
        self.dataset_path = dataset_path
        self.audio_paths = audio_paths
        self.transcripts = transcripts

    def parse_text(self, text):
        return torch.IntTensor(text_to_sequence(text, 'english_cleaner'))

    def get_item(self, index):
        text = self.parse_text(self.transcripts[index])
        mel_spectrogram = self.parse_audio(self.audio_paths[index])

        return text, mel_spectrogram


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
    # TODO


def load_targets(filepath: str, separator: str = '|'):
    target_dict = dict()

    with open(filepath, encoding='utf-8') as f:
        for line in f.readlines():
            audio_path, transcript, _ = line.strip().split(separator)
            target_dict[audio_path] = transcript

    return target_dict
