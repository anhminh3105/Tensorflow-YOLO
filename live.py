from time import time
from argparse import ArgumentParser

import cv2

from yolow import Yolow
from images import display_mess
from utils import parse_args_from_txt


def arg_builder():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-c', '--config',
                            help='path to the model config file',
                            required=True,
                            type=str)
    return arg_parser.parse_args()


def main(args):
    yl = Yolow(args['type'],
                args['model'],
                args['anchors'],
                args['num_classes'],
                (args['width'], args['height']),
                labels=args['labels'])
    cam = cv2.VideoCapture(2)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps_display_interval = 3
    frame_rate = 0
    frame_count = 0
    start_time = time()

    while True:
        _, frame = cam.read()
        # temporary fix for error VIDIOC_DQBUF: Resource temporarily unavailable.
        if frame == None:
            continue
        yl.set_input(frame)
        frame = yl.predict()[0]

        duration = time() - start_time
        if duration >= fps_display_interval:
            frame_rate = round(frame_count/duration)
            start_time = time()
            frame_count = 0
        
        fps_txt = '{} fps'.format(frame_rate)
        frame = display_mess(frame, fps_txt)
        cv2.imshow('YOLOv3 Live', frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = arg_builder()
    main(parse_args_from_txt(args.config))