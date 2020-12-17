# import the necessary packages
from imutils.video import FPS
import numpy as np
import argparse
#from sendsms import sendSMS
from datetime import datetime
import pytz
import cv2
import os
from VideoGetMulti import VideoGetMulti
from CountsPerSec import CountsPerSec

print("start")
video_getter = VideoGetMulti().start()
    # check to see if the output frame should be displayed to our
    # screen
cps = CountsPerSec().start()
while True:
    if 1 > 0:
        frame = video_getter.frame
        frame = cv2.copyMakeBorder(frame, 100,0,0,0, cv2.BORDER_CONSTANT)

        countText = "NoMaskCount: {}  MaskCount: {}".format(video_getter.nomask_count, video_getter.mask_count)
        ratio=video_getter.nomask_count/(video_getter.mask_count+video_getter.nomask_count+0.000001)
        text = countText +" Status: "
        cv2.putText(frame,text, (0, int(50)), cv2.FONT_HERSHEY_SIMPLEX,0.65,[255,255,255], 2)

        if ratio>=0.1 and video_getter.nomask_count>=1:
            warningText = " No Mask !"
            cv2.putText(frame,warningText, (400, int(50)), cv2.FONT_HERSHEY_SIMPLEX,0.65,[26,13,247], 2)
        elif ratio!=0 and np.isnan(ratio)!=True:
            warningText = " Warning !"
            cv2.putText(frame,warningText, (400, int(50)), cv2.FONT_HERSHEY_SIMPLEX,0.65,[0,255,255], 2)
        else:
            warningText = " It's Safe"
            cv2.putText(frame,warningText, (400, int(50)), cv2.FONT_HERSHEY_SIMPLEX,0.65,[0,255,0], 2)

        cv2.rectangle(frame, (video_getter.x, video_getter.y), (video_getter.xw, video_getter.yh), video_getter.color, 3)
        cv2.putText(frame, video_getter.predictText, (video_getter.x, video_getter.y-5), cv2.FONT_HERSHEY_SIMPLEX,0.5, video_getter.color, 1)
        frame = cv2.putText(frame, "{:.0f} iterations/sec".format(cps.countsPerSec()),(10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (70,235,52),2,cv2.LINE_AA)
        cv2.imshow("Frame", frame)
        cps.increment()
        #print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        # show the output frame

        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
