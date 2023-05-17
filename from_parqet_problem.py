import tensorflow as tf
import tensorflow_io as tfio
import pyarrow.parquet as pq

filename = '3855685381.parquet'

# Loading via pyarrow works fine
pqtable = pq.read_table(filename)
xpq = tf.convert_to_tensor(pqtable['x'])
ypq = tf.convert_to_tensor(pqtable['y'])
zpq = tf.convert_to_tensor(pqtable['z'])
print("x via pyarrow into tensor: has nans")
print(xpq)

# Loading using tfio.IOTensor does not work
pqtfio = tfio.IOTensor.from_parquet(filename)
# x and z work
x = pqtfio('x').to_tensor()
z = pqtfio('z').to_tensor()
print("x via pyarrow into tensor: has no nans but seems to fill with last value")
print(x)
# y fails
import ipdb; ipdb.set_trace()
y = pqtfio('y').to_tensor()


# Related issue
