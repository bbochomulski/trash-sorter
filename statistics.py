from keras.models import load_model
import numpy as np
import os
import sklearn.metrics as metrics
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    model = load_model(os.path.join("models", "hindus_v7", "hindus_v7_pass6"))
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
    print(report)

    print('Confusion Matrix')
    cm = metrics.confusion_matrix(test_data_generator.classes, predicted_classes)
    print(cm)
    cm_df = pd.DataFrame(cm,
                         index=['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic'],
                         columns=['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic'])

    plt.figure(figsize=(5, 4))
    plt.title('Confusion Matrix')
    sns.heatmap(cm_df, annot=True)
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.show()