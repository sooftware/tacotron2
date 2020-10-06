# -*- coding: utf-8 -*-
# Soohwan Kim @sooftware
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree

import math
import random
import torch
import threading
from torch.utils.data import Dataset
from tacotron2.data.audio.parser import MelSpectrogramParser
from tacotron2.data.text import text_to_sequence
from tacotron2.utils import logger


class TextMelDataset(Dataset, MelSpectrogramParser):
    """
    Dataset for Text & Mel-Spectrogram matching

    Args:
        audio_paths (list): list of audio_path
        transcripts (list): list of transcript
        feature_extract_by (str): which library to use extract mel spectrogram
        sample_rate (int): sampling rate of audio files
        num_mel_bins (int): number of mel bins
        frame_length_ms (float): frame length milliseconds unit
        frame_shift_ms (float): frame shift milliseconds unit
    """
    def __init__(
            self,
            audio_paths: list,
            transcripts: list,
            feature_extract_by: str,
            sample_rate: int = 22050,
            num_mel_bins: int = 80,
            frame_length_ms: float = 50,
            frame_shift_ms: float = 12.5
    ):
        super(TextMelDataset, self).__init__(feature_extract_by, sample_rate, num_mel_bins, frame_length_ms, frame_shift_ms)
        self.audio_paths = audio_paths
        self.transcripts = transcripts

    def parse_text(self, text):
        return torch.IntTensor(text_to_sequence(text, 'english_cleaner'))

    def get_item(self, index):
        text = self.parse_text(self.transcripts[index])
        mel_spectrogram = self.parse_audio(self.audio_paths[index])

        return text, mel_spectrogram

    def shuffle(self):
        tmp = list(zip(self.audio_paths, self.transcripts))
        random.shuffle(tmp)
        self.audio_paths, self.transcripts = zip(*tmp)

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)


class TextMelDataLoader(threading.Thread):
    def __init__(self, dataset: TextMelDataset, queue, batch_size, thread_id, pad_id):
        threading.Thread.__init__(self)
        self.dataset = dataset
        self.queue = queue
        self.index = 0
        self.batch_size = batch_size
        self.dataset_count = dataset.count()
        self.thread_id = thread_id
        self.pad_id = pad_id

    def create_empty_batch(self):
        seqs = torch.zeros(0, 0, 0)
        targets = torch.zeros(0, 0).to(torch.long)

        seq_lengths = list()
        target_lengths = list()

        return seqs, targets, seq_lengths, target_lengths

    def run(self):
        """ Load data from MelSpectrogramDataset """
        logger.debug('loader %d start' % self.thread_id)

        while True:
            items = list()

            for _ in range(self.batch_size):
                if self.index >= self.dataset_count:
                    break

                text, mel_spectrogram = self.dataset.get_item(self.index)

                if mel_spectrogram is not None:
                    items.append((text, mel_spectrogram))

                self.index += 1

            if len(items) == 0:
                batch = self.create_empty_batch()
                self.queue.put(batch)
                break

            batch = self.collate_fn(items)
            self.queue.put(batch)

        logger.debug('loader %d stop' % self.thread_id)

    def collate_fn(self, batch):
        def seq_length_(p):
            return len(p[0])

        def target_length_(p):
            return len(p[1])

        # sort by sequence length for rnn.pack_padded_sequence()
        batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)

        input_lengths = [len(s[0]) for s in batch]
        target_lengths = [len(s[1]) for s in batch]

        max_seq_sample = max(batch, key=seq_length_)[0]
        max_target_sample = max(batch, key=target_length_)[1]

        max_seq_size = max_seq_sample.size(0)
        max_target_size = len(max_target_sample)

        feat_size = max_seq_sample.size(1)
        batch_size = len(batch)

        inputs = torch.zeros(batch_size, max_seq_size, feat_size)

        targets = torch.zeros(batch_size, max_target_size).to(torch.long)
        targets.fill_(self.pad_id)

        for x in range(batch_size):
            sample = batch[x]
            tensor = sample[0]
            target = sample[1]
            seq_length = tensor.size(0)

            inputs[x].narrow(0, 0, seq_length).copy_(tensor)
            targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

        input_lengths = torch.IntTensor(input_lengths)

        return inputs, targets, input_lengths, target_lengths

    def count(self):
        return math.ceil(self.dataset_count / self.batch_size)


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
