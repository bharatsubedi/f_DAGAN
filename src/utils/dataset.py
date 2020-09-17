import tensorflow as tf
import tensorflow_datasets
import os
from src.utils import config
import numpy as np

NUM_CALLS = tf.data.experimental.AUTOTUNE
NUM_PREFETCH = tf.data.experimental.AUTOTUNE


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.image.resize_with_crop_or_pad(image, [256, 256])
    return image, label


def parse_data(image1_path, image2_path):
    image1 = tf.io.read_file(image1_path)
    image2 = tf.io.read_file(image2_path)

    image1 = tf.image.decode_jpeg(image1, 3)
    image2 = tf.image.decode_jpeg(image2, 3)

    image1 = tf.image.resize_with_crop_or_pad(image1, 28, 28)
    image2 = tf.image.resize_with_crop_or_pad(image2, 28, 28)

    image1 = tf.cast(image1, tf.float32)
    image2 = tf.cast(image2, tf.float32)

    image1 = image1 / 255.0
    image2 = image2 / 255.0
    return image1, image2


def load_data():
    image1_paths = [os.path.join('../../../mnist/train/0', name) for name in os.listdir('../../../mnist/train/0')]
    image2_paths = [os.path.join('../../../mnist/train/0', name) for name in os.listdir('../../../mnist/train/0')]
    np.random.shuffle(image2_paths)
    dataset = tf.data.Dataset.from_tensor_slices((image1_paths, image2_paths))
    dataset = dataset.map(parse_data, num_parallel_calls=NUM_CALLS)
    dataset = dataset.shuffle(config.data_buffer_size)
    dataset = dataset.repeat(config.num_epochs + 1)
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(NUM_PREFETCH)
    return dataset,config.num_epochs
