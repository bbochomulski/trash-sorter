from model import create_model
from training import train_model, create_generators
from testing import test_model, input_photo
from keras.models import load_model

import os


models_dir = 'models'
model_name = 'hindus_1204'
model_path = None
version = None


if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if model_name in os.listdir(models_dir):
    model_dir = os.path.join(models_dir, model_name)
    versions = [version for version in os.listdir(model_dir)]
    if len(versions) > 0:
        versions.sort()
        model_path = os.path.join(model_dir, max(versions))
        version = max(versions).replace(f'{model_name}_pass', '')
    else:
        version = 0
else:
    os.makedirs(os.path.join(models_dir, model_name))
    version = 0

input_shape = (224, 224, 3)

data_dir = 'data-resized'
nb_train_samples = 2000
nb_validation_samples = 400
epochs = 50
batch_size = 100
testing = False


def save_model_to_file(model):
    filename = model_name + '_pass' + str(int(version) + 1)
    print("Model saved as {}".format(filename))
    model.save(os.path.join(models_dir, model_name, filename))
    return filename


def load_model_from_file():
    print("Loaded model: {} pass {}".format(model_name, version))
    return load_model(model_path)


def generate_logfile():
    logfile = os.path.join(models_dir, 'activity.log')
    with open(logfile, 'a') as f:
        f.writelines('\n' + model_name + '\n')
        f.writelines('=' * 20 + '\n')
        f.writelines('nb_train_samples = {} \n'.format(nb_train_samples))
        f.writelines('nb_validation_samples = {} \n'.format(nb_validation_samples))
        f.writelines('epochs = {} \n'.format(epochs))
        f.writelines('batch_size = {} \n'.format(batch_size))
        f.writelines('manual_checked_accuracy = {} \n\n'.format(accuracy))
        with open(os.path.join(models_dir, 'model_history_log.csv'), 'r') as epoch_log:
            f.write(epoch_log.read())
        f.writelines('=' * 20 + '\n')



if not testing:
    if version:
        model = load_model_from_file()
    else:
        model = create_model(model_name, input_shape)

    training_generator, validation_generator = create_generators(data_dir, input_shape, batch_size)

    trained_model = train_model(
        model,
        training_generator,
        validation_generator,
        nb_train_samples,
        nb_validation_samples,
        epochs,
        batch_size
    )
    model_name = save_model_to_file(trained_model)
    accuracy = test_model(trained_model, 50)
else:
    model = load_model_from_file()
    accuracy = test_model(model, 50)

input_photo(model)

if not testing:
    generate_logfile()
