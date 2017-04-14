import cPickle
import numpy as np
import mxnet as mx

def Unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def GetTrainData():
    images = []
    labels = []
    for i in range(1,6):
        dict = Unpickle('./data/data_batch_{0}'.format(i))
        for j in range(10000):
            cur_image = dict['data'][j].reshape(3,32,32)
            cur_image = np.multiply(cur_image,1/255.0)
            cur_label = dict['labels'][j]
            images.append(cur_image)
            labels.append(cur_label)
    return np.array(images),np.array(labels)

def GetTestData():
    images = []
    labels = []
    dict = Unpickle('./data/test_batch')
    for j in range(10000):
        cur_image = dict['data'][j].reshape(3,32,32)
        cur_image = np.multiply(cur_image,1/255.0)
        cur_label = dict['labels'][j]
        images.append(cur_image)
        labels.append(cur_label)
    return np.array(images),np.array(labels)

def Accuracy(label,pred):
    label = label.T.reshape((-1,))
    hit = 0
    total = 0
    for i in range(len(pred)):
        if np.argmax(pred[i]) == int(label[i]):
            hit+=1
        total+=1
    return hit*1.0/total

def NetWork(num_filter):
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('softmax_label')

    conv1 = mx.symbol.Convolution(data=data,kernel=(5,5),num_filter=num_filter,pad=(2,2))
    relu1 = mx.symbol.Activation(data = conv1,act_type='relu')
    pool1 = mx.symbol.Pooling(data=relu1,pool_type='max',kernel=(3,3),stride=(2,2))

    conv2 = mx.symbol.Convolution(data=pool1,kernel=(5,5),num_filter=num_filter,pad=(2,2))
    relu2 = mx.symbol.Activation(data = conv2,act_type='relu')
    pool2 = mx.symbol.Pooling(data=relu2,pool_type='avg',kernel=(3,3),stride=(2,2))

    conv3 = mx.symbol.Convolution(data=pool2,kernel=(5,5),num_filter=num_filter*2,pad=(2,2))
    relu3 = mx.symbol.Activation(data = conv3,act_type='relu')
    pool3 = mx.symbol.Pooling(data=relu3,pool_type='avg',kernel=(3,3),stride=(2,2))

    flat = mx.symbol.Flatten(data=pool3)

    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=64)
    relu4 = mx.symbol.Activation(data=fc1,act_type='relu')

    result = mx.symbol.Reshape(data=label, target_shape=(0,))
    return mx.sym.SoftmaxOutput(data=relu4, label=result, name='softmax')

if __name__ == '__main__':
    train_data,train_label = GetTrainData()
    test_data,test_label = GetTestData()

    batch_size = 8
    data_train = mx.io.NDArrayIter(train_data,train_label,batch_size=batch_size,shuffle=True)
    data_test =mx.io.NDArrayIter(test_data,test_label,batch_size=batch_size,shuffle=True)

    network = NetWork(num_filter=32)

    model = mx.model.FeedForward(symbol=network,ctx=mx.gpu(0),num_epoch=8,learning_rate = 0.1)

    import logging

    head = '%(asctime)-15s %(message)s'

    logging.basicConfig(level=logging.DEBUG, format=head)

    model.fit(X=data_train, eval_data=data_test, eval_metric=Accuracy,

              batch_end_callback=mx.callback.Speedometer(batch_size, 50))


