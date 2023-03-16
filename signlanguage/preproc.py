import tensorflow as tf
import pyarrow.parquet as pq


def read_frame(filename):
    pqtable = pq.read_table(filename)
    x = pqtable['x']
    y = pqtable['y']
    z = pqtable['z']
    coords = tf.stack([x,y,z], axis=1)
    # Reshaping into one row per frame with 1629 features
    # Order: Iterate over types > iterate over landmark index > iterate x y z
    #        Types are ordered alphabetically: face > left hand > pose > right hand
    #        Example: face 0 x, face 0 y, face 0 z, face 1 x, face 1 y, ...
    new_length = int(coords.shape[0] / (1629/3))
    reshaped = tf.reshape(coords, [new_length, 1629])
    return reshaped


if __name__ == "__main__":
    # filename = '/home/jules/handsup/train_landmark_files/16069/3820267556.parquet'
    filename = '3855685381.parquet' # gives errors
    read_frame(filename)
