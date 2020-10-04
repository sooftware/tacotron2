# -*- coding: utf-8 -*-
# Soohwan Kim @sooftware
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree

import torch
import random
import argparse
import warnings
from torch import optim
from tacotron2.criterion.tts_criterion import TextToSpeechCriterion
from tacotron2.model.tacotron2 import Tacotron2
from tacotron2.opts import build_model_opts, build_train_opts
from tacotron2.trainer.supervised_trainer import SupervisedTrainer
from tacotron2.utils import check_envirionment


def train(args):
    SEED = 1011

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = check_envirionment(args.use_cuda)

    # epoch_time_step, trainset_list, validset = split_dataset()
    model = Tacotron2(args)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = TextToSpeechCriterion()

    trainer = SupervisedTrainer()
    model = trainer.train()

    return model


def _get_parser():
    parser = argparse.ArgumentParser(description='Tacotron2')
    parser.add_argument('--mode', type=str, default='train')

    build_model_opts(parser)
    build_train_opts(parser)

    return parser


def main():
    warnings.filterwarnings('ignore')
    parser = _get_parser()
    args = parser.parse_args()
    model = train(args)
    torch.save('checkpoint_last.pt', model)


if __name__ == '__main__':
    main()
