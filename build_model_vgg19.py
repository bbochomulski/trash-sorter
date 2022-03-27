from keras.preprocessing.image import ImageDataGenerator, image_dataset_from_directory
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import optimizers
import tensorflow as tf

img_height, img_width = 384, 512

train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/validation'
nb_train_samples = 500
nb_validation_samples = 100
epochs = 10
batch_size = 50
with tf.device('/cpu:0'):
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    model = Sequential([
        Conv2D(64, kernel_size=(3, 3), padding='same',
               activation='relu', input_shape=input_shape),
        Conv2D(64, kernel_size=(3, 3), padding='same',
               activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(128, kernel_size=(3, 3), padding='same',
               activation='relu'),
        Conv2D(128, kernel_size=(3, 3), padding='same',
               activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(256, kernel_size=(3, 3), padding='same',
               activation='relu'),
        Conv2D(256, kernel_size=(3, 3), padding='same',
               activation='relu'),
        Conv2D(256, kernel_size=(3, 3), padding='same',
               activation='relu'),
        Conv2D(256, kernel_size=(3, 3), padding='same',
               activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(512, kernel_size=(3, 3), padding='same',
               activation='relu'),
        Conv2D(512, kernel_size=(3, 3), padding='same',
               activation='relu'),
        Conv2D(512, kernel_size=(3, 3), padding='same',
               activation='relu'),
        Conv2D(512, kernel_size=(3, 3), padding='same',
               activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(512, kernel_size=(3, 3), padding='same',
               activation='relu'),
        Conv2D(512, kernel_size=(3, 3), padding='same',
               activation='relu'),
        Conv2D(512, kernel_size=(3, 3), padding='same',
               activation='relu'),
        Conv2D(512, kernel_size=(3, 3), padding='same',
               activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(1000, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.003),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
    )

    model.save("vgg19")
