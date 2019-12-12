import numpy as np
import os
import pickle
import tensorflow as tf
from config import FLAGS


def load_facegreyreduxshuffled_set(batch_size, is_training=True):
    path = os.path.join('data', 'facegreyredux')
    if is_training:
        fd = open(os.path.join(path, 'facegreyredux'), 'rb')
        # loaded = np.fromfile(file=fd, dtype=np.uint8)
        loaded = np.asarray(pickle.load(fd))
        trainX = loaded.reshape((57575, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'facegreyreduxcat'), 'rb')
        # loaded = np.fromfile(file=fd, dtype=np.uint8)
        loaded = np.asarray(pickle.load(fd))
        trainY = loaded.reshape((57575)).astype(np.int32)

        data_set = list(zip(trainX,trainY))
        np.random.shuffle(data_set)
        trainX, trainY = list(zip(*data_set))
        trainX = np.asarray(trainX).reshape((57575, 28, 28, 1)).astype(np.float32)
        trainY = np.asarray(trainY).reshape((57575)).astype(np.int32)
        trX = trainX[:52000] / 255.
        trY = trainY[:52000]

        valX = trainX[52000:, ] / 255.
        valY = trainY[52000:]

        num_tr_batch = 52000 // batch_size
        num_val_batch = 5575 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 'facegreyreduxeval'), 'rb')
        loaded = np.asarray(pickle.load(fd))
        trainX = loaded.reshape((10000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'facegreyreduxevalcat'), 'rb')
        loaded = np.asarray(pickle.load(fd))
        trainY = loaded.reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return trainX / 255., trainY, num_te_batch


def create_inputs_norb(path, is_train: bool):
    """Get a batch from the input pipeline.

    Author:
      Ashley Gritzman 15/11/2018
    Args:
      is_train:
    Returns:
      img, lab:
    """
    if is_train:
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_facegreyreduxshuffled_set(FLAGS.batch_size, is_train)
    else:
        trX, trY, num_tr_batch = load_facegreyreduxshuffled_set(FLAGS.batch_size, is_train)

    def generator():
        for e1, e2 in zip(trX, trY):
            yield e1, e2

    capacity = 2000 + 3 * FLAGS.batch_size
    # Create batched dataset
    tf_dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.int32), output_shapes=(tf.TensorShape(list(trX[0].shape)), ())).repeat().shuffle(capacity).batch(batch_size=FLAGS.batch_size, drop_remainder=True).prefetch(1)

    # dataset = input_fn(path, is_train)

    # Create one-shot iterator
    iterator = tf.compat.v1.data.make_one_shot_iterator(tf_dataset)

    img, lab = iterator.get_next()

    output_dict = {'image': img,
                   'label': lab}

    return output_dict
