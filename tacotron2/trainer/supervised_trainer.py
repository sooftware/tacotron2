# -*- coding: utf-8 -*-
# Soohwan Kim @sooftware
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree

import torch
import torch.nn as nn
from tacotron2.data.data_loader import TextMelDataset


class SupervisedTrainer(object):
    """
    The SupervisedTrainer class helps in setting up training framework in a supervised setting.

    Args:
        optimizer (kospeech.optim.optimizer.Optimizer): optimizer for training
        criterion (torch.nn.Module): loss function
        trainset_list (list): list of training datset
        validset (kospeech.data.data_loader.SpectrogramDataset): validation dataset
        num_workers (int): number of using cpu cores
        device (torch.device): device - 'cuda' or 'cpu'
        print_every (int): number of timesteps to print result after
        save_result_every (int): number of timesteps to save result after
        checkpoint_every (int): number of timesteps to checkpoint after
    """
    train_dict = {'loss': [], 'cer': []}
    valid_dict = {'loss': [], 'cer': []}
    train_step_result = {'loss': [], 'cer': []}
    TRAIN_RESULT_PATH = "../data/train_result/train_result.csv"
    VALID_RESULT_PATH = "../data/train_result/eval_result.csv"
    TRAIN_STEP_RESULT_PATH = "../data/train_result/train_step_result.csv"

    def __init__(
            self,
            optimizer: nn.Module,                          # optimizer for training
            criterion: nn.Module,                          # loss function
            trainset_list: list,                           # list of training dataset
            validset: TextMelDataset,                      # validation dataset
            num_workers: int,                              # number of threads
            device: str,                                   # device - cuda or cpu
            print_every: int,                              # number of timesteps to save result after
            save_result_every: int,                        # nimber of timesteps to save result after
            checkpoint_every: int,                         # number of timesteps to checkpoint after
            teacher_forcing_step: float = 0.2,             # step of teacher forcing ratio decrease per epoch.
            min_teacher_forcing_ratio: float = 0.8,        # minimum value of teacher forcing ratio
    ) -> None:
        self.num_workers = num_workers
        self.optimizer = optimizer
        self.criterion = criterion
        self.trainset_list = trainset_list
        self.validset = validset
        self.print_every = print_every
        self.save_result_every = save_result_every
        self.checkpoint_every = checkpoint_every
        self.device = device
        self.teacher_forcing_step = teacher_forcing_step
        self.min_teacher_forcing_ratio = min_teacher_forcing_ratio

    def train(self):
        pass

    def __train_epoches(self):
        pass

    def validate(self):
        pass

    def _save_epoch_result(self):
        pass

    def _save_step_result(self):
        pass
