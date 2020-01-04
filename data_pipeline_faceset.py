import numpy as np
import os
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import rotate, resize
from skimage import exposure
import skimage.io as io



from config import FLAGS


def load_facegreyreduxshuffled_set(batch_size, is_training=True):
    path = os.path.join('data', 'facegreyredux')
    if is_training:
        fd = open(os.path.join(path, 'facegreyredux'), 'rb')
        # loaded = np.fromfile(file=fd, dtype=np.uint8)
        loaded = np.asarray(pickle.load(fd))
        trainX = loaded.reshape((50000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'facegreyreduxcat'), 'rb')
        # loaded = np.fromfile(file=fd, dtype=np.uint8)
        loaded = np.asarray(pickle.load(fd))
        trainY = loaded.reshape((50000)).astype(np.int32)

        data_set = list(zip(trainX,trainY))
        np.random.shuffle(data_set)
        trainX, trainY = list(zip(*data_set))
        trainX = np.asarray(trainX).reshape((50000, 28, 28, 1)).astype(np.float32)
        trainY = np.asarray(trainY).reshape((50000)).astype(np.int32)
        trX = trainX[:40000] / 255.
        trY = trainY[:40000]

        valX = trainX[40000:, ] / 255.
        valY = trainY[40000:]

        num_tr_batch = 40000 // batch_size
        num_val_batch = 10000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        if (FLAGS.flickr):
            fd = open(os.path.join(path, 'flickrsetgreyredux'), 'rb')
            loaded = np.asarray(pickle.load(fd))
            trainX = loaded.reshape((10000, 28, 28)).astype(np.float32) / 255.
        else:
            fd = open(os.path.join(path, 'facegreyreduxeval'), 'rb')
            loaded = np.asarray(pickle.load(fd))
            trainX = loaded.reshape((10000, 28, 28)).astype(np.float32) / 255.

        fd = open(os.path.join(path, 'facegreyreduxevalcat'), 'rb')
        loaded = np.asarray(pickle.load(fd))
        trainY = loaded.reshape((10000)).astype(np.int32)

        rotatedlist = []
        for image in trainX:
            image = rotate(image, FLAGS.rotation, preserve_range=True)
            if(FLAGS.mooney):
                v_min, v_max = np.percentile(image, (49.99999999, 51))
                image = exposure.rescale_intensity(image, in_range=(v_min, v_max))
            rotatedlist.append(image)
            if(len(rotatedlist)==1000):
                I = resize(image.reshape(28, 28), (128, 128))
                io.imsave("rotate" + str(FLAGS.rotation) +  "example.jpg", I, cmap='gray')
        rotatedlist = np.asarray(rotatedlist)
        plt.imshow(rotatedlist[33], cmap='gray')
        plt.show()
        trainX = rotatedlist.reshape((10000, 28, 28, 1)).astype(np.float32)

        return trainX, trainY


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
        trX, trY = load_facegreyreduxshuffled_set(FLAGS.batch_size, is_train)

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
