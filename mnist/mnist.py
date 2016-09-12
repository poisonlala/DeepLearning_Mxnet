import sys

sys.path.insert(0, "../../python")

import mxnet as mx
import numpy as np
import os
from scipy.misc import imread


def Accuracy(label, pred):
    label = label.T.reshape((-1,))
    hit = 0
    total = 0
    for i in range(len(pred)):
        if np.argmax(pred[i]) == int(label[i]):
            hit += 1
        total += 1
    return 1.0 * hit / total


def loadPicture(p):
    images = []
    target = []
    dirs = os.listdir(p)
    for dir in dirs:
        path = os.path.join(p, dir)
        for file in os.listdir(path):
            pic = imread(os.path.join(path, file))
            images.append([decode(pic)])
            target.append(np.array([int(dir)]))
    return np.array(images), np.array(target)


def decode(bmp):
    img = np.multiply(bmp, 1 / 255.0)
    return img


def NetWork():
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('softmax_label')

    fc1 = mx.symbol.FullyConnected(data=data, num_hidden=70)
    act1 = mx.symbol.Activation(data = fc1,act_type='tanh')
    fc2 = mx.symbol.FullyConnected(data=act1, num_hidden=40)
    act2 = mx.symbol.Activation(data=fc2, act_type='tanh')
    fc3 = mx.symbol.FullyConnected(data=act2, num_hidden=10)
    act3 = mx.symbol.Activation(data=fc3, act_type='tanh')

    label = mx.symbol.Reshape(data=label, target_shape=(0,))
    return mx.symbol.SoftmaxOutput(data=act3, label=label, name='softmax')


if __name__ == '__main__':
    net_work = NetWork()

    train_x, train_y = loadPicture('./train/')
    test_x, test_y = loadPicture('./test/')
    batch_size = 8
    data_train = mx.io.NDArrayIter(train_x, train_y, batch_size=batch_size, shuffle=True)
    data_test = mx.io.NDArrayIter(test_x, test_y, batch_size=batch_size, shuffle=True)

    devs = [mx.gpu(i) for i in range(1)]
    model = mx.model.FeedForward(ctx=devs,
                                 symbol=net_work,
                                 num_epoch=2,
                                 learning_rate=0.01,
                                 wd=0.0001,
                                 momentum=0.9)

    import logging

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    model.fit(X=data_train, eval_data=data_test, eval_metric=Accuracy,
              batch_end_callback=mx.callback.Speedometer(batch_size, 50))
