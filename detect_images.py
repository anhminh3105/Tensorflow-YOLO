import time
from argparse import ArgumentParser

from utils import parse_args_from_txt
from images import imwrite


def arg_builder():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-ncs', '--use_ncs',
                            help='whether to infer on the Intels\'NCS\
								 (default to False)',
                            default=False,
                            type=bool)
    arg_parser.add_argument('-c', '--config',
                            help='path to the model config file\
								 (default to data/config/yolov3_full_416.cfg)',
                            default='data/config/yolov3_full_416.cfg',
                            type=str)
    arg_parser.add_argument('-p', '--image_path',
                            help='path to image or directory of images to predict\
                                  (default to data/images/)',
                            default='data/images/',
                            type=str)    
    return arg_parser.parse_args()


args = arg_builder()
model_args = parse_args_from_txt(args.config)
input_sz = model_args['width'], model_args['height']


if args.use_ncs:
	from yolow_ncs import YolowNCS

	yl = YolowNCS(model_args['model_name_path'],
				input_sz, 
				model_args['labels'])
	yl.set_input(args.image_path)

else:
	from yolow import Yolow

	yl = Yolow(model_args['type'],
			model_args['model'],
			model_args['anchors'],
			model_args['num_classes'],
			input_sz,
			model_args['labels'])

	yl.set_input(args.image_path)

start = time.time()
outs = yl.predict()		
print('prediction takes {}s'.format(time.time() - start))
imwrite(outs)

