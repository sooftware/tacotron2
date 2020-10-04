import argparse
import warnings
from tacotron2.opts import build_model_opts, build_train_opts


def train(args):
    # TODO
    pass


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
    train(args)


if __name__ == '__main__':
    main()
