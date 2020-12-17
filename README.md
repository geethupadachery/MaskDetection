# Custom Face Mask Detection using YOLOv3- Application Focused

Real-time inference pipeline from video/camera input using State of art solutions

### Youtube Video Demo: 
[Custom Face Mask Detection(Using OpenCV for Single Thread and Video Acceleration using Multithread](https://www.youtube.com/watch?v=e_6B9EGZv80&feature=youtu.be).

With the help of YoloV3 we can train our own custom face mask detector to detect whether a person is or is not wearing a mask.

### Application Features:

**1. Real time Face Mask Detection - OpenCV Single Threading**
**2. Real time Face Mask Detection- OpenCV Multi-Threading**
**3. Real time Text Alert Notification System using Twilio API** - If the face mask detector application identifies a user that he/she was not wearing a mask, text alerts are sent to the registered phone number of the administrator. It allows the application to run automatically and enforces the wearing of the mask.
**4. Comparison of video inference pipeline for single and multithread**

### Motivation
The COVID-19 pandemic has pushed people aroundthe world to new challenges.
Here, I would like toshow a prototype on how to leverage technology asa mitigation measure against this disease. 
This is an excellent opportunity to put technology at the serviceof humanity.

### Use Cases and Real-life applications meeting the customer requirements:
1. **Airports**:
The Face Mask Detection System can be used at airports to detect travelers without masks. Face data of travelers can be captured in the system at the entrance. If a traveler is found to be without a face mask, the airport authorities can be notified via SMS so that they could take quick action.
2. **Hospitals**
Using Face Mask Detection System, Hospitals can monitor if their staff is wearing masks during their shift or not. If any health worker is found without a mask, they will receive a notification with a reminder to wear a mask. Also, if quarantine people who are required to wear a mask, the system can keep an eye and detect if the mask is present or not and send notification automatically or report to the authorities.
3. **Offices**
The Face Mask Detection System can be used at office premises to detect if employees are maintaining safety standards at work. It monitors employees without masks and sends them a reminder to wear a mask. The reports can be downloaded or sent an email at the end of the day to capture people who are not complying with the regulations or the requirements.

**State of art solutions used:**
For Homework 2, I performed Deep learning-based object detection of face, vehicle and person using Tensorflow2. For final project I am using YOLOV3 for custom face mask detection (using 2 stage detector model).

### Architecture/Design of the Face:	
Utilized a two-stage detector, first a face detector isapplied, to retrieve the face positions. 
Then each face is fed into the second model for detecting“mask” or “no-mask”.
![Architecture/Design of the Face](https://github.com/geethupadachery/MaskDetection/blob/main/detectface.png)

### Steps Followed:
* Enable GPU and check system requirements
* Installation
* Download pretrained YOLOv3 weights
* Run Detections with Darknet and YOLOv3
* Training a Custom YOLOv3 Object Detector
* Prepare data using LabelImg
* Configure the obj.names with the objects todetect
* Configure obj.data file with the number of classesand backup location to store custom weights.
* Generating train.txt and test.txt
* Configuring custom .cfg file for Custom Training
* Train the Custom Object Detector
* Run the Custom Object Detector
* Test trained model on test images
* Test trained model on video
* Evaluate speed of video inference pipeline
* Send SMS alerts using Twilio API when peoplewithout masks are detected.

### Gathering and Labeling a CustomDataset using LabelImg
![labelImg](https://github.com/geethupadachery/MaskDetection/blob/main/lableImg.png)

### Chart showing average loss vs. iterations	
![Chart showing average loss vs. iterations](https://github.com/geethupadachery/MaskDetection/blob/main/graph.png)

### Evaluation Results:
### Test trained model on test images
![Output1](https://github.com/geethupadachery/MaskDetection/blob/main/Output1.png)

![Output2](https://github.com/geethupadachery/MaskDetection/blob/main/output2.png)

![Output3](https://github.com/geethupadachery/MaskDetection/blob/main/output3.png)

![Output4](https://github.com/geethupadachery/MaskDetection/blob/main/Picture1.gif)

### Test trained model on Videos
![OutputVideo1](https://github.com/geethupadachery/MaskDetection/blob/main/test_output.gif)

### Real time SMS Alert System:

**Alert System:** 
It monitors the mask, no-mask counts and has 3 status :

**Safe** : When all people are with mask.

**Warning** : When atleast 1 person is without mask.

**No Mask** : ( + SMS Alert ) When some ratio of people are without mask.

![sms](https://github.com/geethupadachery/MaskDetection/blob/main/sms.PNG)

### Performance Enhancing Strategies:
In order to smooth out the results and to reduce the false alarm of the mask detection I am using two logic:
*	If the face is not found in the frame, the mask detector won’t be applied.
*	I have set a configurable threshold value to send alert based on the NoMask count. For this application purpose, I have put the threshold value for NoMask count as 1. This can be increased according to the user requirements.

### Future Enhancements:
Real-time Monitoring Dashboard
•	A user-friendly website allows the user to see who was not wearing a mask and see the photo or the video captured by the camera. 
•	User can also generate reports to download and integrate with any other third-party integration.

### Comparison of video inference pipeline for single thread and multithread
![comparison](https://github.com/geethupadachery/MaskDetection/blob/main/comparison.png)

**a.	Real time video inference using Single thread:**
The number of frames per second is 1 for single thread video inference pipeline.

**b.	Real time video inference using Multithread:**
The number of frames per second is 23 for multithread video inference pipeline. This is significantly higher than the single thread video inference using OpenCV.

### References:
•	https://pjreddie.com/darknet/yolo/

•	https://arxiv.org/pdf/2003.09093.pdf

•	https://arxiv.org/abs/1804.02767

•	https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/

•	https://medium.com/@ODSC/overview-of-the-yolo-object-detection-algorithm-7b52a745d3e0

•	https://www.analyticsvidhya.com/blog/2018/12/practical-guide-object-detection-yolo-framewor-python/




