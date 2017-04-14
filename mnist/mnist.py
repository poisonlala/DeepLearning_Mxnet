import mxnet as mx
import os
from scipy.misc import imread
import numpy as np

def Accuracy(label,pred):
    label = label.T.reshape((-1,))
    hit = 0
    total = 0
    for i in range(len(pred)):
        if np.argmax(pred[i]) == int(label[i]):
            hit+=1
        total+=1
    return hit*1.0/total

def Load_Image(data_path):
    images=[]
    labels=[]
    dirs= os.listdir(data_path)
    for label in dirs:
        cur_path = os.path.join(data_path,label)
        cur_dir = os.listdir(cur_path)
        for image in cur_dir:
            cur_image = imread(os.path.join(cur_path,image))
            norm_image = np.multiply(cur_image,1/255.0)
            images.append(norm_image)
            labels.append(int(label))
    return np.array(images),np.array(labels)

def Network():
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('softmax_label')

    fc1 = mx.symbol.FullyConnected(data=data,num_hidden=70)
    act1 = mx.symbol.Activation(data = fc1,act_type='tanh')
    fc2 = mx.symbol.FullyConnected(data=act1,num_hidden=40)
    act2 = mx.symbol.Activation(data = fc2,act_type='tanh')
    fc3 = mx.symbol.FullyConnected(data=act2,num_hidden=10)
    act3 = mx.symbol.Activation(data = fc3,act_type='tanh')

    result = mx.symbol.Reshape(data=label,target_shape=(0,))
    return mx.symbol.SoftmaxOutput(data=act3,label=result,name='softmax')

if __name__ == '__main__':
    train_data,train_label = Load_Image(r'./train/')
    test_data,test_label = Load_Image(r'./test/')

    batch_size = 8
    data_train = mx.io.NDArrayIter(train_data,train_label,batch_size=batch_size,shuffle=True)
    data_test = mx.io.NDArrayIter(test_data,test_label,batch_size=batch_size,shuffle=True)

    network = Network()

    model = mx.model.FeedForward(ctx=mx.gpu(0),symbol=network,num_epoch=5,learning_rate=0.01,wd=0.0001,momentum=0.5)

    import logging

    head = '%(asctime)-15s %(message)s'

    logging.basicConfig(level=logging.DEBUG, format=head)

    model.fit(X=data_train, eval_data=data_test, eval_metric=Accuracy,

              batch_end_callback=mx.callback.Speedometer(batch_size, 50))