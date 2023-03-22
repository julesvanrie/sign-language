"""
Module to preprocess the data to shards.
Reads parquet files into a dataset that is saved to shards.
Features are reshaped to tensors with one row per frame.
Labels are integer encoded.


Note
----
Wanted to do this using Tensorflow IO (tfio.IOTensor.from_parquet), but this
turned out to be buggy (also evidenced by several GitHub issues in tfio).
So reverted to suboptimal code using pyarrow in a python function that was
wrapped into a TensorFlow op.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import pyarrow.parquet as pq


DTYPE = tf.float32


def read_frame(filename):
    """
    Reads one individual parquet file into a Tensor with one row per frame.

    Parameters
    ----------
    filename : str or tensorflow.Tensor (scalar)
        String with the path to the parquet file

        Can also accept a scalar tensor which is then decoded to a string.

    Returns
    -------
    tensorflow.Tensor
        Returned shape is one row per frame with 1629 features
        Order: Iterate over types > iterate over landmark index > iterate x y z
            Types are ordered alphabetically: face > left hand > pose > right hand
            Example: face 0 x, face 0 y, face 0 z, face 1 x, face 1 y, ...
    """
    filename = filename.numpy().decode() if isinstance(filename, tf.Tensor) else csvfile
    pqtable = pq.read_table(filename, columns=['x', 'y', 'z'])
    x = pqtable['x']
    y = pqtable['y']
    z = pqtable['z']
    coords = tf.stack([x,y,z], axis=1)
    coords = tf.cast(coords, dtype=DTYPE)
    # Reshaping
    new_length = int(coords.shape[0] / (1629/3))
    reshaped = tf.reshape(coords, [new_length, 1629])
    return reshaped


def encode_labels(labels):
    """
    Encodes the labels to an integer.
    Encoding is done using the json provided by the authors of the Kaggle competition.

    Parameters
    ----------
    labels : Series
        Series containing the labels in string format


    Returns
    -------
    Series
        Series containing np.uint8 encoding of the labels
    """
    mapping = pd.read_json('sign_to_prediction_index_map.json', typ='series', dtype=np.uint8)
    return labels.map(mapping)


def create_full_ds(csvfile):
    """
    Creates a tensorflow.Dataset containing all features in flattened format.

    i.e. one row per frame) and integer encoded labels (following the json
    provided by the authors of the Kaggle competition).

    Parameters
    ----------
    csvfile : str
        Path to the csv file containing filepaths and labels


    Returns
    -------
    tf.data.Dataset
    """
    df = pd.read_csv(csvfile)

    paths = tf.data.Dataset.from_tensor_slices(df.path)
    paths = paths.map(
        lambda path:
            tf.py_function(
                func=read_frame,
                inp=[path],
                Tout=DTYPE,
                name='read_parquet'),
            num_parallel_calls=tf.data.AUTOTUNE)

    labels = encode_labels(df.sign)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    return tf.data.Dataset.zip((paths, labels))


def save_ds_shards(ds):
    """
    Saves the dataset to a number of shards (destination shard is randomly determined).

    Parameters
    ----------
    ds : tensorflow.data.Dataset
        Dataset (can be any dataset)


    Returns
    -------
    None
    """
    NUM_SHARDS = 16
    shard_func = lambda features, labels: tf.random.uniform(shape=[], dtype=tf.int64,
                                                            minval=0, maxval=NUM_SHARDS)
    ds.save("dataset", shard_func=shard_func)


def save_full_ds(csvfile):
    """
    Saves a dataset containing all features in flattened format, in shards.

    Parameters
    ----------
    csvfile : str
        Path to the csv file containing filepaths and labels
    """
    ds = create_full_ds('train.csv')
    save_ds_shards(ds)


if __name__ == "__main__":
    save_full_ds('train.csv')
