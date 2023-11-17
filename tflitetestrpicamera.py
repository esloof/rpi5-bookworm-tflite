import re
import cv2
import numpy as np
import cvzone
import tensorflow.lite as tflite
from PIL import Image
from picamera2 import Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640,480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

model_path='/home/pi/tensorflow/efficientdet_lite0.tflite'
label_path='/home/pi/tensorflow/labels.txt'
def load_labels(label_path):
    r"""Returns a list of labels"""
    with open(label_path) as f:
        labels = {}
        for line in f.readlines():
            m = re.match(r"(\d+)\s+(\w+)", line.strip())
            labels[int(m.group(1))] = m.group(2)
        return labels


def load_model(model_path):
    r"""Load TFLite model, returns a Interpreter instance."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def process_image(interpreter, image, input_index):
    r"""Process an image, Return a list of detected class ids and positions"""
    input_data = np.expand_dims(image, axis=0)  # expand to 4-dim

    # Process
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    # Get outputs
    output_details = interpreter.get_output_details()

    positions = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    classes = np.squeeze(interpreter.get_tensor(output_details[1]['index']))
    scores = np.squeeze(interpreter.get_tensor(output_details[2]['index']))

    result = []

    for idx, score in enumerate(scores):
        if score > 0.5:
            result.append({'pos': positions[idx], 'id': classes[idx]})
    return result


def display_result(result, frame, labels):
    
    for obj in result:
        pos = obj['pos']
        id = obj['id']
        x1 = int(pos[1] * CAMERA_WIDTH)
        x2 = int(pos[3] * CAMERA_WIDTH)
        y1 = int(pos[0] * CAMERA_HEIGHT)
        y2 = int(pos[2] * CAMERA_HEIGHT)
        d=labels[id]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0),1)
        cvzone.putTextRect(frame,f'{d}',(x1,y1),1,1)

    cv2.imshow('Object Detection', frame)


if __name__ == "__main__":

    model_path = 'efficientdet_lite0.tflite'
    label_path = 'labels.txt'
    interpreter = load_model(model_path)
    labels = load_labels(label_path)

    input_details = interpreter.get_input_details()

    # Get Width and Height
    input_shape = input_details[0]['shape']
    height = input_shape[1]
    width = input_shape[2]

    # Get input index
    input_index = input_details[0]['index']

    # Process Stream
    while True:
        im= picam2.capture_array()
        im=cv2.flip(im,-1)
        image = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        image = image.resize((width, height))

        top_result = process_image(interpreter, image, input_index)
        display_result(top_result, im, labels)

        key = cv2.waitKey(1)
        if key == 27:  # esc
            break


    cv2.destroyAllWindows()
