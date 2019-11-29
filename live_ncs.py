from time import time

import cv2
## FOR PICAMERA HARDWARE
# from picamera.array import PiRGBArray
# from picamera import PiCamera

from yolow_ncs import YolowNCS
from images import display_mess
from argparse import ArgumentParser
from utils import parse_args_from_txt


def arg_builder():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-c', '--config',
                            help='path to the model config file\
                                 (default to data/config/ncs/yolov3_full_416.cfg)',
                            default='data/config/ncs_yolov3_full_416.cfg',
                            type=str)
    arg_parser.add_argument('-r', '--requests',
                            help='the number of requests for \
                                  multi-request pool feature (default to 4)',
                            default=2,
                            type=int)
    return arg_parser.parse_args()


def main(args):
    model_args = parse_args_from_txt(args.config)
    input_size = (model_args['width'], model_args['height'])
    yl_ncs = YolowNCS(model_args['model_name_path'],
                    input_size, 
                    model_args['labels'],
                    args.requests)
    fps_display_interval = 3
    frame_rate = 0
    frame_count = 0
    async_mode = True
    ## FOR PICAMERA HARDWARE
    # cam = PiCamera()
    # cam.resolution = (720, 480)
    # cam.framerate = 30
    # rawCapture = PiRGBArray(cam)
    ##
    ## FOR OTHER CASES
    cam = cv2.VideoCapture(-1)
    ##

    start_time = time()
    ## FOR PICAMERA HARDWARE
    # for frame in cam.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    #     frame = frame.array
    ##
    ## FOR OTHER CASES
    while cam.isOpened():
        _, frame = cam.read()
    ##
        # temporary fix for error VIDIOC_DQBUF: Resource temporarily unavailable.
        if frame == None:
            continue
        yl_ncs.set_input(frame)
        # start = time()
        # in async mode, some beginnig frames causes nan values which breaks the programme
        # skip it for now
        try:
            frame = yl_ncs.predict(async_mode=async_mode)[0]
        except:
            pass

        duration = time() - start_time
        if duration > fps_display_interval:
            frame_rate = round(frame_count/duration)
            start_time = time()
            frame_count = 0
        
        mode = "async" if async_mode else "sync"
        status = '{} fps - mode: {}'.format(frame_rate, mode)
        frame = display_mess(frame, status)
        cv2.imshow('NCS YOLOv3 Live', frame)
        frame_count += 1

        key = cv2.waitKey(1)
        if key & 0xFF==ord('q'):
            break

        if key & 0xFF==ord('m'):
            async_mode = not async_mode
    ## FOR OTHER CASES
    cam.release()
    ##
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(arg_builder())
    