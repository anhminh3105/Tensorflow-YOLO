from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
from time import time
from yolow_ncs import YolowNCS
from imager import *
import io

imer = Imager()
yl_ncs= YolowNCS()
camera = PiCamera()
camera.resolution = (1024, 720)
camera.framerate = 30
rawCapture = PiRGBArray(camera)
frame_interval = 1
fps_display_interval = 3
frame_rate = 0
frame_count = 0
start_time = time()

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = frame.array
    if frame_count % frame_interval == 0:
        imer.imset(frame)
        input_list = imer.ncs_preprocess()
        start = time()
        pred_list = yl_ncs.predict(input_list)
        print('this frame takes {:.2f}s to process.'.format(time() - start))  
        imer.visualise_preds(pred_list)
        
    duration = time() - start_time
    if duration > fps_display_interval:
        frame_rate = float(frame_count/duration)
        start_time = time()
        frame_count = 0

    fps_txt = '{:.2f} fps'.format(frame_rate)    
    frame = imer.display_fps(fps_txt)
    cv2.imshow('NCS YOLOv3 Live', frame)
    frame_count += 1
    rawCapture.truncate(0)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cv2.destroyAllWindows()
