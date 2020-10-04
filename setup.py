# -*- coding: utf-8 -*-
# Soohwan Kim @sooftware
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree

from distutils.core import setup

setup(
    name='Tacotron2',
    version='0.0',
    install_requires=[
        'torch>=1.4.0',
        'librosa >= 0.7.0',
        'numpy',
        'pandas'
    ]
)
