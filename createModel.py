import os


from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, Lambda, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import cv2
IMG_SIZE = 64
category = ['Uninfected', 'Parasitized']
def get_train_data(direct):
    data = []
    for labels in category:
        path = os.path.join(direct, labels)
        class_num = category.index(labels)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([new_array, class_num])
            except Exception as e:
                print(e)
    return np.array(data,dtype=object)
new_data = get_train_data("cell_images/cell_images/")
X = []
y = []
for feature, label in new_data:
    X.append(feature)
    y.append(label)
    
from tensorflow.keras.utils import to_categorical
y = to_categorical(y, num_classes = 2)

X = np.array(X)
y = np.array(y)

X = X.reshape(-1, 64, 64, 3)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_generator = ImageDataGenerator(rescale = 1/255,
                                     zoom_range = 0.3,
                                     horizontal_flip = True,
                                     rotation_range = 30)

test_generator = ImageDataGenerator(rescale = 1/255)

train_generator = train_generator.flow(np.array(X_train),
                                       y_train,
                                       batch_size = 64,
                                       shuffle = False)

test_generator = test_generator.flow(np.array(X_test),
                                     y_test,
                                     batch_size = 64,
                                     shuffle = False)
vg19 = VGG19(input_shape=[IMG_SIZE, IMG_SIZE] + [3], weights="imagenet", include_top=False)

for layer in vg19.layers:
    layer.trainable = False

x = Flatten()(vg19.output)
prediction = Dense(len(category), activation="softmax")(x)
model = Model(inputs=vg19.input, outputs=prediction)


model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")
history = model.fit(train_generator,
                    steps_per_epoch = len(X_train)//64,
                    epochs = 1,
                    validation_data=test_generator,
                    validation_steps=len(X_test)//64)
model.evaluate(test_generator, steps=len(X_test)//64)


import matplotlib.pyplot as plt
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, "b", label="trainning accuracy")
plt.plot(epochs, val_accuracy, "r", label="validation accuracy")
plt.legend()
plt.show()

plt.plot(epochs, loss, "b", label="trainning loss")
plt.plot(epochs, val_loss, "r", label="validation loss")
plt.legend()
plt.show()
y_pred = model.predict(test_generator)
y_pred
y_pred = np.argmax(y_pred, axis=1)
y_pred
model.save("model2.h5")

