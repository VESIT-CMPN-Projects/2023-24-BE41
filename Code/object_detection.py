import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import tensorflow as tf
from tensorflow import keras

class Lanes():
    def _init_(self):
        self.recent_fit = []
        self.avg_fit = []

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
lanes = Lanes()
# Load your pre-trained model
def load_object_detection_model():
    prototxt_path = "MobileNetSSD_deploy.prototxt"
    model_path = "MobileNetSSD_deploy.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    return net

def load_lane_detection_model():
    lanemodel = keras.models.load_model('model.h5')
    return lanemodel

# Object detection pipeline
def object_detection_pipeline(frame, model):
    if frame is not None and not frame.size == (0, 0):
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        model.setInput(blob)
        detections = model.forward()

        for key in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, key, 2]

            if confidence > 0.2:
                idx = int(detections[0, 0, key, 1])

                (h, w) = frame.shape[:2]
                box = detections[0, 0, key, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Print label, confidence, and coordinates
                label = "Class: {}, Confidence: {:.2f}%".format(CLASSES[idx], confidence * 100)
                coordinates = "Coordinates: ({}, {}, {}, {})".format(startX, startY, endX, endY)
                # print(label, coordinates)

                color = (0, 255, 0)  # Green color
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                #cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)


    return frame

def road_lines(image, model):
    small_img = cv2.resize(image, (160, 80))  # Resize image using OpenCV
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    prediction = model.predict(small_img)[0] * 255

    lanes.recent_fit.append(prediction)
   
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis=0)
    
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    lane_image = cv2.resize(lane_drawn, (1280, 720))  # Resize the lane image

    # Convert both images to np.uint8
    image = image.astype(np.uint8)
    lane_image = lane_image.astype(np.uint8)

    # Apply addWeighted with output data type specified as np.uint8
    result = cv2.addWeighted(image, 1, lane_image, 1, 0, dtype=cv2.CV_8U)

    return result