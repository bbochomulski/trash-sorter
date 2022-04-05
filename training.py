from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger


def create_generators(data_dir, img_shape, batch_size):
    """
    Tworzenie generatorów danych treningowych i walidacyjnych
    :param data_dir: folder z danymi
    :param img_shape: kształt obrazka (szerokosc x wysokosc)
    :param batch_size: ilość obiektów w jednej mini-batch
    :return: generator danych treningowych i walidacyjnych
    """
    # tworzenie danych augumentacji
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # tworzenie generatora danych treningowych
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_shape[0], img_shape[1]),
        batch_size=batch_size,
        class_mode='binary'
    )

    # tworzenie generatora dla danych walidacyjnych
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_shape[0], img_shape[1]),
        batch_size=batch_size,
        class_mode='binary'
    )
    return train_generator, validation_generator


def train_model(model, train_generator, validation_generator, nb_train_samples, nb_validation_samples, epochs, batch_size):
    """
    Trenowanie modelu

    :param model: obiekt modelu do treningu
    :param train_generator: generator danych treningowych
    :param validation_generator: generator danych walidacyjnych
    :param nb_train_samples: ilość obiektów w danych treningowych
    :param nb_validation_samples: ilość obiektów w danych walidacyjnych
    :param epochs: ilość epok
    :param batch_size: ilość obiektów w jednej mini-batch
    :return: trained model
    """

    csv_logger = CSVLogger("models/model_history_log.csv", append=False)

    model.fit(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[csv_logger],
        verbose=1
    )

    return model
