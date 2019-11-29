import sys
from time import time
from ctypes import c_bool
from threading import Thread
from argparse import ArgumentParser
from multiprocessing import Process, Queue, Event, Value

import cv2

from images import display_mess
from yolow_ncs import YolowNCS
from utils import parse_args_from_txt

def arg_builder():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-c', '--config',
                            help='path to the model config file\
                                 (default to data/config/ncs_yolov3_full_416.cfg)',
                            default='data/config/ncs_yolov3_full_416.cfg',
                            type=str)
    arg_parser.add_argument('-n', '--sticks',
                            help='the number of Movidius NCSes \
                                  to run inference on (default to 2).',
                            default=2,
                            type=int)
    arg_parser.add_argument('-r', '--requests',
                            help='the number of requests for \
                                  multi-request feature (default to 4)',
                            default=4,
                            type=int)
    return arg_parser.parse_args()

def live_job(model_args, input_buffer, output_buffer, async_mode, quit_event):
    fps_display_interval = 1
    frame_rate = 0
    frame_count = 0
    processing_frame_rate = 0
    processing_frame_count = 0
    pred_list = None
    ## FOR PICAMERA HARDWARE
    # cam = PiCamera()
    # cam.resolution = (720, 480)
    # cam.framerate = 30
    # rawCapture = PiRGBArray(cam)
    ##
    ## FOR OTHER CASES
    cam = cv2.VideoCapture(0)
    ##
    start_time = time()
    ## FOR PICAMERA HARDWARE
    # for frame in cam.capture_continuous(rawCapture, format="bgr", use_video_port=True)
    #    frame = frame.array
    ##
    ## FOR OTHER CASES
    while not quit_event.is_set():
        # read from camera.
        _, frame = cam.read()
    ##
        #temporary fix for error VIDIOC_DQBUF: Resource temporarily unavailable.
        if frame == None:
            continue

        # makes space in the buffer if it's full.
        if input_buffer.full():
            input_buffer.get()
        # put frame to the processing buffer.
        input_buffer.put(frame)
        # wait to get the value.
        while output_buffer.empty():
            continue
        frame = output_buffer.get()
        processing_frame_count += 1

        duration = time() - start_time
        if duration > fps_display_interval:
            frame_rate = round(frame_count/duration)
            frame_count = 0
            processing_frame_rate = round(processing_frame_count/duration)
            processing_frame_count = 0
            start_time = time()
            
        mode = "async" if async_mode.value else "sync"
        status = '{} fps (detection: {} fps) - mode: {}'.format(frame_rate, processing_frame_rate, mode)
        frame = display_mess(frame, status)
        cv2.imshow('NCS YOLOv3 Live', frame)
        frame_count += 1

        key = cv2.waitKey(1)
        if key & 0xFF==ord('q'):
            quit_event.set()
            break

        if key & 0xFF==ord('m'):
            async_mode.value = not async_mode.value

    cam.release()
    cv2.destroyAllWindows()


def ncs_worker_thread(yl_ncs, input_buffer, output_buffer, async_mode, quit_event):
    while not quit_event.is_set():
        if not input_buffer.empty():
            frame = input_buffer.get()
            yl_ncs.set_input(frame)
            # In async mode, early frames causes NaN values which breaks the programme,
            # skip it for now.
            try:
                frame = yl_ncs.predict(async_mode=async_mode)[0]
            except:
                pass           
            output_buffer.put(frame)

def infer_job(model_args, sticks, requests, input_buffer, output_buffer, async_mode, quit_event):
    threads = []
    for _ in range(sticks):
        th = Thread(target=ncs_worker_thread,
                    args=(YolowNCS(model_args['model_name_path'], 
                                   (model_args['width'], model_args['height']),
                                   model_args['labels'],
                                   requests), 
                    input_buffer, 
                    output_buffer, 
                    async_mode, 
                    quit_event))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()

def main(args):
    model_args = parse_args_from_txt(args.config)
    processes = []
    input_buffer = Queue(10)
    output_buffer = Queue()
    quit_event = Event()
    async_mode = Value(c_bool, True)
    # create networks for each stick assign it to a separate process.
    p = Process(target=infer_job,
                args=(model_args, args.sticks, args.requests, input_buffer, output_buffer, async_mode, quit_event),
                daemon=True)
    p.start()
    processes.append(p)
    # for dev_id in range(stick_num):
    #     p = Process(target=ncs_worker_thread,
    #                 args=(YolowNCS(), input_buffer, output_buffer), daemon=True)
    #     p.start()
    #     processes.append(p)
    #start live job.
    p = Process(target=live_job,
                args=(model_args, input_buffer, output_buffer, async_mode, quit_event),
                daemon=True)
    p.start()
    processes.append(p)

    while not quit_event.is_set():
        pass

    for p in processes:
        p.terminate()

if __name__ == "__main__":
    main(arg_builder())
