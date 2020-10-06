# -*- coding: utf-8 -*-
# Soohwan Kim @sooftware
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree

import math
import random
from tacotron2.data.data_loader import TextMelDataset
from tacotron2.utils import logger


def load_targets(filepath: str, separator: str = '|'):
    audio_paths = list()
    transcripts = list()

    with open(filepath, encoding='utf-8') as f:
        for line in f.readlines():
            audio_path, transcript, _ = line.strip().split(separator)
            audio_paths.append(audio_path)
            transcripts.append(transcript)

    return audio_paths, transcripts


def split_dataset(args):
    logger.info("split dataset start !!")

    trainset_list = list()
    audio_paths, transcripts = load_targets(args.metadata_path, separator='|')

    train_num = math.ceil(len(audio_paths) * (1 - args.valid_ratio))
    total_time_step = math.ceil(len(audio_paths) / args.batch_size)
    valid_time_step = math.ceil(total_time_step * args.valid_ratio)
    train_time_step = total_time_step - valid_time_step

    train_num_per_worker = math.ceil(train_num / args.num_workers)

    # audio_paths & script_paths shuffled in the same order
    # for seperating train & validation
    tmp = list(zip(audio_paths, transcripts))
    random.shuffle(tmp)
    audio_paths, transcripts = zip(*tmp)

    # seperating the train dataset by the number of workers
    for idx in range(args.num_workers):
        train_begin_idx = train_num_per_worker * idx
        train_end_idx = min(train_num_per_worker * (idx + 1), train_num)

        trainset_list.append(
            TextMelDataset(
                audio_paths=audio_paths[train_begin_idx:train_end_idx],
                transcripts=transcripts[train_begin_idx:train_end_idx],
                feature_extract_by=args.feature_extract_by,
                sample_rate=args.sample_rate,
                num_mel_bins=args.num_mel_bins,
                frame_length_ms=args.frame_length_ms,
                frame_shift_ms=args.frame_shift_ms
            )
        )

    validset = TextMelDataset(
        audio_paths=audio_paths[train_num:],
        transcripts=transcripts[train_num:],
        feature_extract_by=args.feature_extract_by,
        sample_rate=args.sample_rate,
        num_mel_bins=args.num_mel_bins,
        frame_length_ms=args.frame_length_ms,
        frame_shift_ms=args.frame_shift_ms
    )

    logger.info("split dataset complete !!")
    return train_time_step, trainset_list, validset