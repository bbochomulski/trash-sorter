from keras.preprocessing import image
import random
import os
import numpy as np
from tqdm.auto import tqdm
from resizeimage import resizeimage


classes = {
        0: "cardboard",
        1: "glass",
        2: "metal",
        3: "paper",
        4: "plastic"
}


def verdict(output):
    output = list(output[0])
    output_class = classes[output.index(max(output))]
    return output_class


def prep_img():
    chosen_class = list(classes.values())[random.randint(0, len(classes)-1)]
    path = f'data-resized/{chosen_class.capitalize()}'
    filename = random.choice(os.listdir(path))
    img_pred = image.load_img(os.path.join(path, filename))
    img_pred = image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred, axis=0)
    return img_pred, chosen_class


def test_model(model, n_images):
    passed = 0
    for _ in tqdm(range(n_images)):
        img_pred, chosen_class = prep_img()
        output = model.predict(img_pred)
        if verdict(output) == chosen_class:
            passed += 1

    print("\nAccuracy")
    print(f"{passed} of {n_images} passed")
    print("{:.2f} %".format(passed / n_images * 100))
    return passed / n_images * 100


def input_photo(model, img=None):
    images_list = os.listdir('test') if img is None else [img]
    for img_pred in images_list:
        if img is None:
            path = os.path.join('test', img_pred)
            img_pred = image.load_img(path)
        img_pred = resizeimage.resize_contain(img_pred, [224, 224])
        img_pred = img_pred.convert("RGB")
        img_pred = image.img_to_array(img_pred)
        img_pred = np.expand_dims(img_pred, axis=0)
        output = model.predict(img_pred)
        if len(images_list) == 1:
            return verdict(output)
        else:
            print("{} is a {}".format(path, verdict(output)))
