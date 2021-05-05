import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# def test():
#     # Initialization of tensors
#     w = tf.constant(4, shape=(1, 1), dtype=tf.float32)
#     x = tf.constant(5, shape=(3, 3), dtype=tf.float32)
#     y = tf.zeros((2, 3))
#     z = tf.eye(3)
#     a = tf.random.normal((3, 3), mean=0, stddev=1)
#     b = tf.random.uniform((7, 15), minval=0, maxval=1)
#
#     # Mathematical operations
#
#     result = tf.subtract(x, a)
#     result2 = tf.multiply(x, a)
#     result3 = tf.tensordot(x, a, axes=1)
#     print(result)
#     print(result2)
#     print(result3)
#     print(b * 10)
#
#     # Indexing
#
#     val = tf.constant([0, 1, 1, 2, 3, 1, 2, 3])
#     print(val[:])
#     print(val[1:])
#     print(val[1:3])
#     print(val[::2])
#     i = tf.constant([0, 3])
#     indy_boi = tf.gather(val, i)
#     print(indy_boi)
#     x = tf.constant([[1, 2],
#                     [3, 4],
#                     [5, 6]])
#     print(x[0, :])
#     print(x[0:2, :])
#
#     # Reshaping
#
#     x = tf.range(9)
#     print(x)
#
#     x = tf.reshape(x, (3, 3))
#     print(x)
#
#     x = tf.transpose(x, perm=[1, 0])
#     print(x)


def neural_test():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape)
    print(y_train.shape)
    x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0
    model = keras.Sequential(
        [
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(10),
        ]
    )

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(lr=0.001),
        metrics=["accuracy"],
    )

    model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
    model.evaluate(x_test, y_test, batch_size=32, verbose=2)


neural_test()
