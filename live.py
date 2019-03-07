import cv2
from time import time
from yolow import Yolow
from imager import *

imer = Imager()
yl= Yolow()
cam = cv2.VideoCapture(0)
frame_interval = 1
fps_display_interval = 5
frame_rate = 0
frame_count = 0
start_time = time()

while True:
    _, frame = cam.read()

    if frame_count % frame_interval == 0:
        imer.imset(frame)
        input_list = imer.preproces()
        pred_list = yl.predict(input_list)
        imer.visualise_preds(pred_list)

    duration = time() - start_time
    if duration >= fps_display_interval:
        frame_rate = int(frame_count/duration)
        start_time = time()
        frame_count = 0
        
    frame = imer.display_fps(frame_rate)
    cv2.imshow('YOLOv3 Live', frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()


