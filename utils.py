from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import load_model
import numpy as np

YOLO_CLASSES = []

def crop_image(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Crops an image to the target width and height"""
    image_width, image_height = image.size

    remaining_width = image_width - target_width
    remaining_height = image_height - target_height

    return image.crop(((remaining_width // 2),
                       (remaining_height // 2),
                       (image_width - (remaining_width // 2)),
                       (image_height - (remaining_height // 2))))


# Help create or add text to an image
def text_to_image(text, img_width, img_height, base_image=None):
    if base_image:
        img = base_image
    else:
        img = Image.new('RGB', (img_width, img_height), color = 'black')
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 30)

    draw.text((0,0), str(text), fill=(255,255,255), font=font)
    return img

# Load Yolo model into memory
def getTinyYoloV3Model():
    model = load_model('yolov3-tiny.h5')
    return model

# Load the 80 Yolo classes into memory
def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

# This code inspired from: # https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/
def tiny_yolo_decode_classes(netout):
    global YOLO_CLASSES
    if len(YOLO_CLASSES) == 0:
        YOLO_CLASSES = load_classes('coco.names.txt')
        print('Loaded classes: ',len(YOLO_CLASSES))
    nb_box = 3
    grid_h, grid_w = netout.shape[:2]
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))

    obj_thresh = 0.6

    # Below is the black magic part
    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh


    nb_class = netout.shape[-1] - 5 # => 80
    classes = []

    for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if objectness <= obj_thresh: 
                continue
            # first 4 elements are x, y, w, and h
            #x, y, w, h = netout[int(row)][int(col)][b][:4]
            #x = (col + x) / grid_w # center position, unit: image width
            #y = (row + y) / grid_h # center position, unit: image height
            #w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            #h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
            # last elements are class probabilities
            classes_80_proba = netout[int(row)][col][b][5:]
            classes.append(YOLO_CLASSES[np.argmax(classes_80_proba)])
            #box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
            #boxes.append(box)
    return classes
    