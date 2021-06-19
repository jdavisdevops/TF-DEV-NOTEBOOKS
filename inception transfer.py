import os
import pathlib
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import Adam, RMSprop

# os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
tf.distribute.OneDeviceStrategy(device="/gpu:0")
policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.experimental.set_policy(policy)
datadir = pathlib.Path(
    r"C:\Users\josephdavis\Desktop\Tensorflow notebooks\cats-v-dogs\training"
)
valdir = pathlib.Path(
    r"C:\Users\josephdavis\Desktop\Tensorflow notebooks\cats-v-dogs\testing"
)

AUTOTUNE = tf.data.AUTOTUNE

logdir = pathlib.Path.cwd() / "logs"

if len(os.listdir(logdir)) > 5:
    shutil.rmtree(logdir)
image_count = len(list(datadir.glob("*/*.jpg")))
batch_size = 128
img_height = 150
img_width = 150
list_ds = tf.data.Dataset.list_files(str(datadir / "*/*.jpg"), shuffle=False)
val_ds = tf.data.Dataset.list_files(str(valdir / "*/*.jpg"), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
num_train_files = len(list_ds)
num_val_files = len(val_ds)
class_names = np.array(sorted([item.name for item in datadir.glob("*")]))


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == class_names
    return tf.argmax(one_hot)


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    return img  # tf.image.resize(img, [img_height, img_width])


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    #     return tf.data.Dataset.from_tensors((img, label))
    return img, label


train_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

data_augmentation = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomTranslation(.2, .2),
        layers.experimental.preprocessing.RandomRotation(0.2, interpolation='bilinear'),
        layers.experimental.preprocessing.RandomWidth(0.01, interpolation="bilinear"),
        layers.experimental.preprocessing.RandomHeight(0.01, interpolation="bilinear"),
        layers.experimental.preprocessing.RandomContrast(0.2),
        layers.experimental.preprocessing.RandomZoom(0.1),
    ]
)

resize_and_rescale = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.Resizing(img_height, img_width),
        layers.experimental.preprocessing.Rescaling(1.0 / 255),
    ]
)


def prepare(ds, shuffle=False, augment=False):
    # Resize and rescale all datasets.
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=AUTOTUNE)

    ds.cache()

    if shuffle:
        ds = ds.shuffle(1000)

    # Batch all datasets
    ds = ds.batch(batch_size)

    if augment:
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        ).cache()

    # Use buffered prefecting on all datasets
    return ds.prefetch(buffer_size=AUTOTUNE)


train_ds = prepare(train_ds, shuffle=True, augment=True)
val_ds = prepare(val_ds)

pre_trained_model = InceptionV3(
    input_shape=(img_height, img_width, 3), include_top=False, weights="imagenet"
)

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer("mixed7")
print("last layer output shape: ", last_layer.output_shape)
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation="sigmoid")(x)

model = Model(pre_trained_model.input, x)

model.compile(
    optimizer=RMSprop(lr=0.001), loss="binary_crossentropy", metrics=["accuracy"]
)

logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=logs, histogram_freq=1, profile_batch="500,520"
)

callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", mode="auto", patience=4, restore_best_weights=True
)

history = model.fit(
    train_ds,
    epochs=50,
    verbose=1,
    validation_data=val_ds,
    callbacks=[callback, tboard_callback],
    #     steps_per_epoch=int(num_train_files / batch_size),
    #     validation_steps=int(num_val_files / batch_size),
)

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

for value in history.history:
    print(value, history.history[value])

epochs = range(len(acc))

plt.plot(epochs, acc, "r", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend(loc=0)
plt.figure()

plt.show()
