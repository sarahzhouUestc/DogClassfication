# encoding=utf-8
"""
拼接特征，训练一个2层的神经网络，然后在测试集上进行预测
"""
import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet.gluon import nn

import h5py
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

ctx = mx.gpu()
def accuracy(output, labels):
    return nd.mean(nd.argmax(output, axis=1) == labels).asscalar()

def evaluate(net, data_iter):
    loss, acc, n = 0., 0., 0.
    steps = len(data_iter)
    for data, label in data_iter:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        output = net(data)
        acc += accuracy(output, label)
        loss += nd.mean(softmax_cross_entropy(output, label)).asscalar()
    return loss/steps, acc/steps

# 载入之前计算的特征向量
with h5py.File('features.h5', 'r') as f:
    features_vgg = np.array(f['vgg'])
    features_resnet = np.array(f['resnet'])
    features_densenet = np.array(f['densenet'])
    features_inception = np.array(f['inception'])
    labels = np.array(f['labels'])
features_resnet = features_resnet.reshape(features_resnet.shape[:2])
features_inception = features_inception.reshape(features_inception.shape[:2])
features = np.concatenate([features_resnet, features_densenet, features_inception], axis=-1)

# 利用ArrayDataset和DataLoader构建迭代器
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2)

dataset_train = gluon.data.ArrayDataset(nd.array(X_train), nd.array(y_train))
dataset_val = gluon.data.ArrayDataset(nd.array(X_val), nd.array(y_val))

batch_size = 128
data_iter_train = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)
data_iter_val = gluon.data.DataLoader(dataset_val, batch_size)

# 构建模型
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dropout(0.5))
    net.add(nn.Dense(120))

net.initialize(ctx=ctx)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 1e-4, 'wd': 1e-5})
net

# 训练50代
for epoch in range(50):
    train_loss = 0.
    train_acc = 0.
    steps = len(data_iter_train)
    for data, label in data_iter_train:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)

        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)

        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)

    val_loss, val_acc = evaluate(net, data_iter_val)
    print("Epoch %d. loss: %.4f, acc: %.2f%%, val_loss %.4f, val_acc %.2f%%" % (
        epoch + 1, train_loss / steps, train_acc / steps * 100, val_loss, val_acc * 100))

# 载入测试集特征向量
with h5py.File('features_test.h5', 'r') as f:
    features_vgg_test = np.array(f['vgg'])
    features_resnet_test = np.array(f['resnet'])
    features_densenet_test = np.array(f['densenet'])
    features_inception_test = np.array(f['inception'])
features_resnet_test = features_resnet_test.reshape(features_resnet_test.shape[:2])
features_inception_test = features_inception_test.reshape(features_inception_test.shape[:2])

features_test = np.concatenate([features_resnet_test, features_densenet_test, features_inception_test], axis=-1)

# 利用模型进行预测并输出到pred.csv
output = nd.softmax(net(nd.array(features_test).as_in_context(ctx))).asnumpy()
df = pd.read_csv('sample_submission.csv')

for i, c in enumerate(df.columns[1:]):
    df[c] = output[:,i]

df.to_csv('pred.csv', index=None)

