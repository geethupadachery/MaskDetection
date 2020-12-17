from threading import Thread
import cv2
import numpy as np
from imutils.video import FPS
import numpy as np
import argparse
from datetime import datetime
import pytz
import os

class VideoGetMulti:

    def __init__(self):
        self.x = 0
        self.y = 0
        self.xw = 0
        self.yh = 0
        self.color = ""
        self.predictText = ""
        self.mask_count = 0
        self.nomask_count = 0
        self.stream = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        # it will get the time zone
        # of the specified location
        PST = pytz.timezone('US/Pacific')
        

        # construct the argument parse and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-y", "--yolo", required=True,help="base path to YOLO directory")
        ap.add_argument("-i", "--input", type=str, default="",help="path to (optional) input video file")
        ap.add_argument("-o", "--output", type=str, default="",help="path to (optional) output video file")
        ap.add_argument("-d", "--display", type=int, default=1,help="whether or not output frame should be displayed")
        ap.add_argument("-c", "--confidence", type=float, default=0.45,help="minimum probability to filter weak detections")
        ap.add_argument("-t", "--threshold", type=float, default=0.3,help="threshold when applyong non-maxima suppression")
        ap.add_argument("-u", "--use-gpu", type=bool, default=0,help="boolean indicating if CUDA GPU should be used")
        args = vars(ap.parse_args())

        # load the class labels our YOLO model was trained on
        labelsPath = os.path.sep.join([args["yolo"], "obj.names"])
        LABELS = open(labelsPath).read().strip().split("\n")
        #print('LABELS', LABELS)

        # initialize a list of colors to represent each possible class label(red and green)
        COLORS = [[0,255,0],[0,0,255]]

        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join([args["yolo"], "yolov3_face_mask.weights"])
        configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

        # load our YOLO object detector trained on mask dataset
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        # check if we are going to use GPU
        if args["use_gpu"]:
            # set CUDA as the preferable backend and target
            print("[INFO] setting preferable backend and target to CUDA...")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


        # initialize the width and height of the frames in the video file
        W = None
        H = None

        # initialize the video stream and pointer to output video file, then
        # start the FPS timer
        print("[INFO] accessing video stream...")
        vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
        writer = None
        next_frame_towait=5 #for sms
        # loop over frames from the video file stream
        while True:
            # read the next frame from the file
            (self.grabbed, self.frame) = vs.read()
            # if the frame was not grabbed, then we have reached the end
            # of the stream
            if not self.grabbed:
                break
            # if the frame dimensions are empty, grab them
            if W is None or H is None:
                (H, W) = self.frame.shape[:2]
            # construct a blob from the input frame and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes
            # and associated probabilities
            blob = cv2.dnn.blobFromImage(self.frame, 1 / 255.0, (864, 864),swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)
            # initialize our lists of detected bounding boxes, confidences,
            # and class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []
            # loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability)
                    # of the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > args["confidence"]:
                        # scale the bounding box coordinates back relative to
                        # the size of the image, keeping in mind that YOLO
                        # actually returns the center (x, y)-coordinates of
                        # the bounding box followed by the boxes' width and
                        # height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        # use the center (x, y)-coordinates to derive the top
                        # and and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        # update our list of bounding box coordinates,
                        # confidences, and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
            # apply NMS to suppress weak, overlapping
            # bounding boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],args["threshold"])

            #Add top-border to frame to display stats
            border_size=100
            border_text_color=[255,255,255]
            filtered_classids=np.take(classIDs,idxs)
            nomask_count=(filtered_classids==1).sum()
            mask_count=(filtered_classids==0).sum()
            self.mask_count = mask_count
            self.nomask_count = nomask_count

            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():

                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1]+border_size)
                    (w, h) = (boxes[i][2], boxes[i][3])
                    # draw a bounding box rectangle and label on the image
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    self.x, self.y = (x, y)
                    self.xw, self.yh =  (x + w, y + h)
                    self.color = color
                    #cv2.rectangle(self.frame, (x, y), (x + w, y + h), color, 1)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    self.predictText = text
                    #cv2.putText(self.frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1)
