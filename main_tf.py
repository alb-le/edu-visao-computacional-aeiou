import cv2
import numpy as np
import tensorflow as tf
from time import sleep

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# 1. Load the TensorFlow SavedModel and get the default serving signature
model = tf.saved_model.load('model.savedmodel')
infer = model.signatures["serving_default"]

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)
palavra = ""
while True:
    # Grab the webcamera's image.
    ret, image = camera.read()
    if not ret:
        break

    # Resize the raw image into (224-height, 224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the model's input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # --- TensorFlow SavedModel Inference ---
    # Convert numpy array to a TensorFlow tensor
    input_tensor = tf.constant(image)
    
    # Run prediction via the signature (returns a dictionary of output tensors)
    outputs = infer(input_tensor)
    
    # Extract the first output tensor and convert it back to a numpy array
    prediction = list(outputs.values())[0].numpy()
    # ---------------------------------------

    index = np.argmax(prediction)
    letra_reconhecida = class_names[index][2:]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", letra_reconhecida, end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.

    if keyboard_input == 27:
        break
    if keyboard_input == ord(' '):
        palavra += letra_reconhecida
        print(f"PALAVRA ATUAL:\n{palavra}")
        sleep(1)  # Espera um pouco antes de continuar para evitar múltiplas adições rápidas

camera.release()
cv2.destroyAllWindows()
print(f"PALAVRA ESCRITA:\n{palavra}")
