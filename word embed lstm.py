import io
import os
import re
import shutil
import string
import tensorflow as tf
from tensorflow.keras import Sequential, mixed_precision
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tarfile
from pathlib import Path
from tensorboard.plugins import projector
from tensorflow import keras

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
mixed_precision.set_global_policy('mixed_float16')

cwd = Path.cwd()
datadir = cwd / 'aclImdb'

logdir = cwd / 'logs'

# if logdir.exists():
#     shutil.rmtree(logdir)
#     print('removed logdir')

if datadir.exists():
    pass
else:
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    dataset = tf.keras.utils.get_file(fname="aclImdb_v1", origin=url, untar=True, cache_dir='.', cache_subdir='')

    for file in os.listdir(os.getcwd()):
        if file.endswith('.tar.gz'):
            tar = tarfile.open(file, 'r:gz')
            tar.extractall()
            tar.close()

train_dir = datadir / 'train'

# remove_dir = os.path.join(train_dir, 'unsup')
remove_dir = train_dir / 'unsup'
if remove_dir.exists():
    shutil.rmtree(remove_dir)
else:
    pass

batch_size = 1024
seed = 123

train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', batch_size=batch_size, validation_split=0.2, subset='training', seed=seed
)

val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', batch_size=batch_size, validation_split=0.2, subset='validation', seed=seed
)

# for text_batch, label_batch in train_ds.take(1):
#     for i in range(5):
#         print(label_batch[i].numpy(), text_batch.numpy()[i])

autotune = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=autotune)
val_ds = val_ds.cache().prefetch(buffer_size=autotune)

embedding_layer = tf.keras.layers.Embedding(1000, 5)


# Create a custom standardization function to strip HTML break tags '<br />'.
@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation), '')


# Vocabulary size and number of words in a sequence.
vocab_size = 10000
sequence_length = 100

# Use the text vectorization layer to normalize, split, and map strings to
# integers. Note that the layer uses the custom standardization defined above.
# Set maximum_sequence length as all samples are not of the same length.
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (no labels) and call adapt to build the vocabulary.
text_ds = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)

embedding_dim = 16

model = Sequential([
    vectorize_layer,
    Embedding(vocab_size, embedding_dim, name='embedding'),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1)
])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

model_dir = cwd / 'models'
vec_model = model_dir / 'vec_model'
if model_dir.exists():
    model = keras.models.load_model(vec_model)
else:
    Path.mkdir(model_dir)
    model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[tensorboard_callback])
    model.save(vec_model)

tensor_weights = tf.Variable(model.get_layer('embedding').get_weights()[0][1:])
weights = model.get_layer('embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()

out_m = io.open(cwd / 'metadata.tsv', 'w+', encoding='utf-8')

for index, word in enumerate(vocab):
    if index == 0:
        continue  # skip 0, it's padding.


    out_m.write(word + "\n")
out_m.close()

checkpoint = tf.train.Checkpoint(embedding=tensor_weights)
checkpoint.save(logdir / 'embedding.ckpt')

config = projector.ProjectorConfig()
embedding = config.embeddings.add()

embedding.tensor_name = 'embedding/.ATTRIBUTES/VARIABLE_VALUE'
embedding.metadata_path = r'C:\Users\josephdavis\Desktop\Tensorflow Test\metadata.tsv'
projector.visualize_embeddings(logdir, config)

tb = 'tensorboard --logdir logs'
os.system(tb)
