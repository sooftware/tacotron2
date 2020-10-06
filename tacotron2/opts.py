# -*- coding: utf-8 -*-
# Soohwan Kim @sooftware
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree

from argparse import ArgumentParser


def build_preprocess_opts(parser: ArgumentParser):
    pass


def build_model_opts(parser: ArgumentParser):
    # encoder arguments
    group = parser.add_argument_group('Model')
    group.add_argument('--vocab_size', '-vocab_size',
                       type=int, default=10,
                       help='size of character vocab')
    group.add_argument('--embedding_dim', '-embedding_dim',
                       type=int, default=512,
                       help='dimension of character embedding layer')
    group.add_argument('--encoder_lstm_dim', '-encoder_lstm_dim',
                       type=int, default=256,
                       help='dimension of rnn hidden state vector')
    group.add_argument('--num_encoder_lstm_layers', '-num_encoder_lstm_layers',
                       type=int, default=1,
                       help='number of encoder lstm layers')
    group.add_argument('--conv_dropout_p', '-conv_dropout_p',
                       type=float, default=0.5,
                       help='dropout probability of convolution layer')
    group.add_argument('--num_encoder_conv_layers', '-num_encoder_conv_layers',
                       type=int, default=3,
                       help='number of encoder`s convolution layers')
    group.add_argument('--encoder_conv_kernel_size', '-encoder_conv_kernel_size',
                       type=int, default=1,
                       help='size of convolution layer`s kernel')
    group.add_argument('--encoder_lstm_bidirectional', '-encoder_lstm_bidirectional',
                       action='store_true', default=False,
                       help='if True, becomes a bidirectional encoder (defulat: False)')
    group.add_argument('--device', '-device',
                       type=str, default='cuda',
                       help='size of character vocab')

    # decoder arguments
    group.add_argument('--num_mel_bins', '-num_mel_bins',
                       type=int, default=80,
                       help='number of mel filters')
    group.add_argument('--prenet_dim', '-prenet_dim',
                       type=int, default=256,
                       help='dimension of prenet')
    group.add_argument('--decoder_lstm_dim', '-decoder_lstm_dim',
                       type=int, default=1024,
                       help='dimension of decoder lstm network')
    group.add_argument('--attn_lstm_dim', '-attn_lstm_dim',
                       type=int, default=1024,
                       help='dimension of attention lstm network')
    group.add_argument('--attn_dim', '-attn_dim',
                       type=int, default=128,
                       help='dimension of attention layer')
    group.add_argument('--location_conv_filter_size', '-location_conv_filter_size',
                       type=int, default=32,
                       help='size of location convolution filter')
    group.add_argument('--location_conv_kernel_size', '-location_conv_kernel_size',
                       type=int, default=31,
                       help='size of location convolution kernel')
    group.add_argument('--prenet_dropout_p', '-prenet_dropout_p',
                       type=float, default=0.5,
                       help='dropout probability of prenetb')
    group.add_argument('--attn_dropout_p', '-attn_dropout_p',
                       type=float, default=0.1,
                       help='dropout probability of attention network')
    group.add_argument('--decoder_dropout_p', '-decoder_dropout_p',
                       type=float, default=0.1,
                       help='dropout probability of decoder network')
    group.add_argument('--max_decoding_step', '-max_decoding_step',
                       type=int, default=1000,
                       help='max decoding step')
    group.add_argument('--stop_threshold', '-stop_threshold',
                       type=float, default=0.5,
                       help='stop threshold')

    # postnet arguments
    group.add_argument('--postnet_dim', '-postnet_dim',
                       type=int, default=512,
                       help='dimension of postnet')
    group.add_argument('--num_postnet_conv_layers', '-num_postnet_conv_layers',
                       type=int, default=5,
                       help='number of convolution layers')
    group.add_argument('--postnet_conv_kernel_size', '-postnet_conv_kernel_size',
                       type=int, default=5,
                       help='size of convolution kernel')
    group.add_argument('--postnet_dropout_p', '-postnet_dropout_p',
                       type=int, default=0.5,
                       help='probability of dropout')


def build_train_opts(parser: ArgumentParser):
    group = parser.add_argument_group('Train')
    group.add_argument('--teacher_forcing_ratio', '-teacher_forcing_ratio',
                       type=int, default=1.0,
                       help='probability of teacher forcing (default: 1.0)')
    group.add_argument('--lr', '-lr',
                       type=int, default=1.0,
                       help='learning rate for training')
    group.add_argument('--weight_decay', '-weight_decay',
                       type=int, default=1.0,
                       help='weight decay for optimizer')