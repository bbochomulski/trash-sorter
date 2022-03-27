from keras.models import load_model
from keras import backend as k
import numpy as np
from keras.preprocessing import image
import os
import random
from tqdm.auto import tqdm

img_height, img_width = 384, 512

classes = {
      0: "cardboard",
      1: "glass",
      2: "metal",
      3: "paper",
      4: "plastic"
}

if k.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = load_model("hindus_v2")
print("\nHindus_v3")
print("Hyper Intelligent Neural DUmp Sorter\n")


def prep_img():
    chosen_class = list(classes.values())[random.randint(0, len(classes)-1)]
    path = f'dataset/validation/{chosen_class.capitalize()}'
    number_files = len(os.listdir(path))
    img_pred = image.load_img(f'{path}/{chosen_class.lower()}{random.randint(1, number_files)}.jpg', target_size=(img_width, img_height))
    img_pred = image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred, axis=0)
    return img_pred, chosen_class


def verdict(output, expect):
    output = list(output[0])
    # print(f"Prediction: {output}")
    # print("Result: ", end='')
    output_class = classes[output.index(max(output))]
    # print(f'{output_class} \t expected: {expect}')
    # if output_class == expect:
    #     print("Verdict: Passed")
    # else:
    #     print("Verdict: Failed")
    return output_class


def perform_test(n):
    passed = 0
    for _ in tqdm(range(n)):
        img = prep_img()
        predicted = (verdict(model.predict(img[0]), img[1]))
        expected = (img[1])
        if predicted == expected:
            passed += 1
    return passed


nr_of_images = 100
result = perform_test(nr_of_images)

print("\nAccuracy")
print(f"{result} of {nr_of_images}")
print("{:.2f} %".format(result/nr_of_images*100))
