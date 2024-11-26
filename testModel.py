import tensorflow as tf
import numpy as np
import cv2
import os

IMG_SIZE = 64
CATEGORIES = ['Uninfected', 'Parasitized']


def prepare(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image file not found: {filepath}")

    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if img_array is None:
        raise ValueError(f"Failed to load image: {filepath}")

    img_array = img_array.astype(np.float32) / 255.0  # Normalize to [0, 1]
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return np.expand_dims(new_array, axis=0)  # Add batch dimension


def predict_image(model, filepath):
    try:
        img = prepare(filepath)
        prediction = model.predict(img, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]
        return CATEGORIES[predicted_class], confidence
    except Exception as e:
        return f"Error: {str(e)}", None


# Load the model
model_path = "model.h5"
if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

model = tf.keras.models.load_model(model_path)

# List of images to predict
images = [
    "test/uninf.png",
    "test/inf.png"
]

# Make predictions
for img_path in images:
    result, confidence = predict_image(model, img_path)
    if confidence is not None:
        print(f"Image: {img_path}")
        print(f"Prediction: {result}")
        print(f"Confidence: {confidence:.2f}")
    else:
        print(result) 
    print()