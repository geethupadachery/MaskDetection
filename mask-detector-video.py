# import the necessary packages
from imutils.video import FPS
import numpy as np
import argparse
from sendsms import sendSMS
from datetime import datetime 
import pytz 
import cv2
import os
from CountsPerSec import CountsPerSec
from facedetection import facedetection

# it will get the time zone  
# of the specified location 
IST = pytz.timezone('US/Pacific')


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

# Loading model for face detection
faceNet, faceln = facedetection().load_face_detection_model(args["yolo"])

# load the class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "obj.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label(red and green)
COLORS = [[0,255,0],[0,0,255]]

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3_face_mask.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on mask dataset
print("[INFO] loading model for mask detection...")
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
fps = FPS().start()
cps = CountsPerSec().start()
# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break
    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (864, 864),swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    # # First detect faces in the captured image
    # faces = facedetection().detect_faces(blob, args["confidence"], args["threshold"], H, W)
    
    # # Checking if there are any faces detected. Will continue with mask detection only if faces are detected.
    # #print("faces ", faces)
    # if faces < 1:
    #     print("no faces detected")
    #     continue
    
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
    frame = cv2.copyMakeBorder(frame, border_size,0,0,0, cv2.BORDER_CONSTANT)
    #calculate count values
    filtered_classids=np.take(classIDs,idxs)
    nomask_count=(filtered_classids==1).sum()
    mask_count=(filtered_classids==0).sum()
    #display count
    text = "NoMaskCount: {}  MaskCount: {}".format(nomask_count, mask_count)
    cv2.putText(frame,text, (0, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.65,border_text_color, 2)
    #display status
    text = "Status:"
    cv2.putText(frame,text, (W-200, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.65,border_text_color, 2)
    ratio=nomask_count/(mask_count+nomask_count+0.000001)

    
    if ratio>=0.1 and nomask_count>=1:
        text = "No Mask!"
        cv2.putText(frame,text, (W-100, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.65,[26,13,247], 2)
        if fps._numFrames>=next_frame_towait: #to send danger sms again,only after skipping few seconds
            msg="**Face Mask Alert** \n"
            #msg+="Camera ID: C001 %0A%0A"            
            msg+="Status: No Mask Detected !!! \n"
            msg+="No Mask Count: "+str(nomask_count)+" \n "
            msg+="Mask Count: "+str(mask_count)+" \n "
            datetime_ist = datetime.now(IST) 
            msg+="Alert sent on : "+datetime_ist.strftime('%Y-%m-%d %H:%M:%S %Z')
            sendSMS(msg)
            print('Text Message Sent')
            next_frame_towait=fps._numFrames+(5*25)
        
    elif ratio!=0 and np.isnan(ratio)!=True:
        text = "Warning !"
        cv2.putText(frame,text, (W-100, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.65,[0,255,255], 2)

    else:
        text = "It's Safe"
        cv2.putText(frame,text, (W-100, int(border_size-50)), cv2.FONT_HERSHEY_SIMPLEX,0.65,[0,255,0], 2)
    

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1]+border_size)
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1)

    # check to see if the output frame should be displayed to our
    # screen
    if args["display"] > 0:
        # show the output frame
        frame = cv2.putText(frame, "{:.0f} iterations/sec".format(cps.countsPerSec()),(10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (70,235,52),2,cv2.LINE_AA)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    # if an output video file path has been supplied and the video
    # writer has not been initialized, do so now
    if args["output"] != "" and writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 24,(frame.shape[1], frame.shape[0]), True)
    # if the video writer is not None, write the frame to the output
    # video file
    if writer is not None:
        writer.write(frame)
    # update the FPS counter
    fps.update()
    cps.increment()
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
