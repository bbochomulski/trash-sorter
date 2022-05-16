from keras.models import load_model
import numpy as np
import os
import sklearn.metrics as metrics
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_statistics(model_path):
    model = load_model(model_path)
    test_generator = ImageDataGenerator()
    test_data_generator = test_generator.flow_from_directory(
        os.path.join('data-resized'),
        target_size=(224, 224),
        batch_size=32,
        shuffle=False)
    test_steps_per_epoch = np.math.ceil(test_data_generator.samples / test_data_generator.batch_size)

    predictions = model.predict_generator(test_data_generator, steps=test_steps_per_epoch)
    predicted_classes = np.argmax(predictions, axis=1)

    true_classes = test_data_generator.classes
    class_labels = list(test_data_generator.class_indices.keys())

    report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
    print('Classification Report')
    file_name = 'summary_{}.txt'.format(model_path.split("\\")[-2])
    if 'pass1' in model_path:
        with open(os.path.join("statistics", model_path.split("\\")[-2], file_name), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

    print('Confusion Matrix')
    cm = metrics.confusion_matrix(test_data_generator.classes, predicted_classes)
    print(cm)
    cm_df = pd.DataFrame(cm,
                         index=['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic'],
                         columns=['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic'])

    plt.figure(figsize=(5, 4))
    plt.title(model_path.split("\\")[-1])
    sns.heatmap(cm_df, annot=True, fmt='g')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.show()
    file_name = 'confusion_matrix_{}.png'.format(model_path.split("\\")[-1])
    plt.savefig(os.path.join("statistics", model_path.split("\\")[-2], file_name))

    return report


if __name__ == "__main__":
    model_name = "hindus_newdataset"
    path = os.path.join("models", model_name)
    os.makedirs(os.path.join("statistics", model_name), exist_ok=True)
    with open(os.path.join("statistics", model_name, f"{model_name}.txt"), 'a+') as f:
        for model in os.listdir(path):
            f.write(model + '\n' + create_statistics(os.path.join(path, model)) + '\n\n')