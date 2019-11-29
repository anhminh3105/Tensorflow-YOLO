import time
from argparse import ArgumentParser

from yolow import Yolow
from utils import parse_args_from_txt


def arg_builder():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-c', '--config',
                            help='path to the model config file',
                            required=True,
                            type=str)
    return arg_parser.parse_args()


start = time.time()
args = arg_builder()
args = parse_args_from_txt(args.config)
print('freezing {} model...'.format(args['model']))
yl = Yolow(args['type'],
            args['model'],
            args['anchors'],
            args['num_classes'],
            (args['width'], args['height']),
            labels=args['labels'])
print('total time: {}'.format(time.time() - start))

