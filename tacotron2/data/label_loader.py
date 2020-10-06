# -*- coding: utf-8 -*-
# Soohwan Kim @sooftware
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree


def load_targets(filepath: str, separator: str = '|'):
    audio_paths = list()
    transcripts = list()

    with open(filepath, encoding='utf-8') as f:
        for line in f.readlines():
            audio_path, transcript, _ = line.strip().split(separator)
            audio_paths.append(audio_path)
            transcripts.append(transcript)

    return audio_paths, transcripts