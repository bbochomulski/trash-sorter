from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.metrics import Precision, Recall
import os


def create_model(model_name, input_shape):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(128, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(128, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((3, 3)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    if os.path.exists(model_name):
        print("Loading model...")
        model = load_model(model_name)
    else:
        print("Creating model...")
        print(model.summary())

    return model
