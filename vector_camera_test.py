import asyncio
from concurrent.futures import CancelledError
import os
import sys

import numpy as np

##from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions # First version uses MobileNet
from tensorflow.keras.preprocessing.image import img_to_array

import utils
import anki_vector

from PIL import Image, ImageDraw

# Width of images passed to the network
IMAGE_WIDTH: int = 416 # 416 for YoloV3 net, 224 for MobileNetV2
# Height of images passed to the network
IMAGE_HEIGHT: int = 416

screen_dimensions = anki_vector.screen.SCREEN_WIDTH, anki_vector.screen.SCREEN_HEIGHT

#model = MobileNetV2(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3), alpha=1.0, include_top=True, weights='imagenet', pooling=None, classes=1000)
model = utils.getTinyYoloV3Model()

def make_prediction(robot):
    while True:
        camera_image = robot.camera.latest_image.raw_image
                    
        # Crop the image to reduce the complexity of the network
        cropped_image = utils.crop_image(camera_image, IMAGE_WIDTH, IMAGE_HEIGHT)
        # Convert image to an array with shape (image_width, image_height, 1)
        image = img_to_array(cropped_image)
        # Normalize the image data
        image = image.astype("float") / 255.0
        # Expand array shape to add an axis to denote the number of images fed as input
        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image)[0] 
        predicted_classes = utils.tiny_yolo_decode_classes(prediction[0])
        
        #predicted_classes = decode_predictions(prediction.reshape(1, 1000)) # mobilenet
        #print('Classes:',predicted_classes)
        #max_filtered_result = [x for x in predicted_classes[0] if x[2] > 0.6] # mobilenet
        
        max_filtered_result = [x for x in predicted_classes]
        if len(max_filtered_result) > 0:
            #max_filtered_result = max_filtered_result[0] # Taking the best prediction
            print("Main class: ",max_filtered_result)
            
            # Display prediction on vector screen + captured image
            screen_data = anki_vector.screen.convert_image_to_screen_data(utils.text_to_image(max_filtered_result, screen_dimensions[0], screen_dimensions[1], base_image=utils.crop_image(cropped_image, screen_dimensions[0], screen_dimensions[1])))
            
            robot.screen.set_screen_with_image_data(screen_data, 1)
            # Vector can say prediction, funny at first, annoying in a second time :p 
            #robot.behavior.say_text(max_filtered_result[1])


with anki_vector.Robot(None) as robot:
    robot.camera.init_camera_feed()
    make_prediction(robot)

    