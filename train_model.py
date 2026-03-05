import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

data = []
labels = []

dataset_path = "dataset"

classes = ["Normal","Fighting"]

for label, category in enumerate(classes):

    path = os.path.join(dataset_path, category)

    for video in os.listdir(path):

        video_path = os.path.join(path, video)

        cap = cv2.VideoCapture(video_path)

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.resize(frame,(64,64))
            frame = frame / 255.0

            data.append(frame)
            labels.append(label)

        cap.release()

X = np.array(data)
y = np.array(labels)

model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(2,activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X,y,epochs=10)

model.save("violence_model.h5")

print("Model saved successfully")