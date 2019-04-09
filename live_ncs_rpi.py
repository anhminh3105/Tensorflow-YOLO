from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
from time import time
from yolow_ncs import YolowNCS
from imager import *

imer = Imager()
yl_ncs= YolowNCS()
camera = PiCamera()
camera.resolution = (720, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera)
fps_display_interval = 3
frame_rate = 0
frame_count = 0
start_time = time()
async_mode = False


for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = frame.array
    imer.imset(frame)
    input_list = imer.ncs_preprocess()
    start = time()
    pred_list = yl_ncs.predict(input_list, async_mode=async_mode)
    print('this frame takes {:.2f}s to process.'.format(time() - start))  
    imer.visualise_preds(pred_list)
        
    duration = time() - start_time
    if duration > fps_display_interval:
        frame_rate = round(frame_count/duration)
        start_time = time()
        frame_count = 0

    mode = "async" if async_mode else "sync"
    fps_txt = '{} fps - mode: {}'.format(frame_rate, mode)
    frame = imer.display_fps(fps_txt)
    cv2.imshow('NCS YOLOv3 Live', frame)
    frame_count += 1
    rawCapture.truncate(0)

    key = cv2.waitKey(1)
    if key & 0xFF==ord('q'):
        break

    if key & 0xFF==ord('m'):
        async_mode = not async_mode

cv2.destroyAllWindows()
