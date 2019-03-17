import cv2
from time import time
from yolow import Yolow
from imager import *

imer = Imager()
yl= Yolow()
cam = cv2.VideoCapture(0)
fps_display_interval = 3
frame_rate = 0
frame_count = 0
start_time = time()

while True:
    _, frame = cam.read()
    imer.imset(frame)
    input_list = imer.preproces()
    pred_list = yl.predict(input_list)
    imer.visualise_preds(pred_list)

    duration = time() - start_time
    if duration >= fps_display_interval:
        frame_rate = round(frame_count/duration)
        start_time = time()
        frame_count = 0
    
    fps_txt = '{} fps'.format(frame_rate)
    frame = imer.display_fps(fps_txt)
    cv2.imshow('YOLOv3 Live', frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
