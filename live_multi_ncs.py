import sys, cv2
from imager import *
from time import time
from yolow_ncs import YolowNCS
from argparse import ArgumentParser
from multiprocessing import Process, Queue, Event, Value
from threading import Thread
from ctypes import c_bool
def arg_builder():
    arg_parser = ArgumentParser()
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

def live_job(input_buffer, output_buffer, async_mode, quit_event):
    imer = Imager()
    cam = cv2.VideoCapture(0)
    fps_display_interval = 1
    frame_rate = 0
    frame_count = 0
    processing_frame_rate = 0
    processing_frame_count = 0
    start_time = time()
    pred_list = None
    
    while not quit_event.is_set():
        mode = "async" if async_mode.value else "sync"
        # read from camera.
        _, frame = cam.read()
        imer.imset(frame)
        # makes space in the buffer if it's full.
        if input_buffer.full():
            input_buffer.get()
        # put frame to the processing buffer.
        input_list = imer.ncs_preprocess()
        input_buffer.put(input_list)
        # wait to get the value.
        if not output_buffer.empty():
            pred_list = output_buffer.get()
            processing_frame_count += 1
        if pred_list is not None:
            imer.visualise_preds(pred_list) 

        duration = time() - start_time
        if duration > fps_display_interval:
            frame_rate = round(frame_count/duration)
            frame_count = 0
            processing_frame_rate = round(processing_frame_count/duration)
            processing_frame_count = 0
            start_time = time()
        
        fps_txt = '{} fps (detection: {} fps) - mode: {}'.format(frame_rate, processing_frame_rate, mode)
        frame = imer.display_fps(fps_txt)
        cv2.imshow('NCS YOLOv3 Live', frame)
        frame_count += 1

        key = cv2.waitKey(1)
        if key & 0xFF==ord('q'):
            quit_event.set()
            break

        if key & 0xFF==ord('m'):
            async_mode.value = not async_mode.value
            print("Switched to {} mode".format(mode))

    cam.release()
    cv2.destroyAllWindows()


def ncs_worker_thread(yl_ncs, input_buffer, output_buffer, async_mode, quit_event):
    while not quit_event.is_set():
        if not input_buffer.empty():
            input_list = input_buffer.get()
            pred_list = yl_ncs.predict(input_list, async_mode=async_mode.value)
            output_buffer.put(pred_list)

def infer_job(num_sticks, num_requests, input_buffer, output_buffer, async_mode, quit_event):
    threads = []
    for _ in range(num_sticks):
        th = Thread(target=ncs_worker_thread,
                    args=(YolowNCS(num_requests=num_requests), input_buffer, output_buffer, async_mode, quit_event))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()

def main(args):
    num_sticks = args.sticks
    num_requests = args.requests
    processes = []
    input_buffer = Queue(10)
    output_buffer = Queue()
    quit_event = Event()
    async_mode = Value(c_bool, True)
    # create networks for each stick assign it to a separate process.
    p = Process(target=infer_job,
                args=(num_sticks, num_requests, input_buffer, output_buffer, async_mode, quit_event),
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
                args=(input_buffer, output_buffer, async_mode, quit_event),
                daemon=True)
    p.start()
    processes.append(p)

    while not quit_event.is_set():
        pass

    for p in processes:
        p.terminate()

if __name__ == "__main__":
    main(arg_builder())
