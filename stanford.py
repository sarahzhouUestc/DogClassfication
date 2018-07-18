# encoding=utf-8
"""
使用stanford数据集来训练
"""
import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.data import vision
from mxnet.gluon.model_zoo import vision as models
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import h5py
import os

import matplotlib.pyplot as plt
ctx = [mx.gpu(i) for i in range(4)] # 如果是单卡，需要修改这里
df = pd.read_csv('sample_submission.csv')
synset = list(df.columns[1:])

# 载入数据集
from glob import glob
n = len(glob('Images/*/*.jpg'))
X_224 = nd.zeros((n, 3, 224, 224))
X_299 = nd.zeros((n, 3, 299, 299))
y = nd.zeros((n,))

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

for i, file_name in tqdm(enumerate(glob('Images/*/*.jpg')), total=n):
    img = cv2.imread(file_name)
    img_224 = ((cv2.resize(img, (224, 224))[:, :, ::-1] / 255.0 - mean) / std).transpose((2, 0, 1))
    img_299 = ((cv2.resize(img, (299, 299))[:, :, ::-1] / 255.0 - mean) / std).transpose((2, 0, 1))

    X_224[i] = nd.array(img_224)
    X_299[i] = nd.array(img_299)
    y[i] = synset.index(file_name.split('/')[1][10:].lower())
    nd.waitall()

# 定义得到预训练模型特征的函数
def get_features(model_name, data_iter):
    net = models.get_model(model_name, pretrained=True, ctx=ctx)
    features = []
    for data in tqdm(data_iter):
        # 并行预测数据，如果是单卡，需要修改这里
        for data_slice in gluon.utils.split_and_load(data, ctx, even_split=False):
            feature = net.features(data_slice)
            feature = gluon.nn.Flatten()(feature)
            features.append(feature.as_in_context(mx.cpu()))
        nd.waitall()

    features = nd.concat(*features, dim=0)
    return features

# 计算几个预训练模型输出的特征并拼接起来
batch_size = 128

data_iter_224 = gluon.data.DataLoader(gluon.data.ArrayDataset(X_224), batch_size=batch_size)
data_iter_299 = gluon.data.DataLoader(gluon.data.ArrayDataset(X_299), batch_size=batch_size)

model_names = ['inceptionv3', 'resnet152_v1']
features = []
for model_name in model_names:
    if model_name == 'inceptionv3':
        features.append(get_features(model_name, data_iter_299))
    else:
        features.append(get_features(model_name, data_iter_224))

features = nd.concat(*features, dim=1)
data_iter_train = gluon.data.DataLoader(gluon.data.ArrayDataset(features, y), batch_size, shuffle=True)

# 定义一些函数
def build_model():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.BatchNorm())
        net.add(nn.Dense(1024))
        net.add(nn.BatchNorm())
        net.add(nn.Activation('relu'))
        net.add(nn.Dropout(0.5))
        net.add(nn.Dense(120))

    net.initialize(ctx=ctx)
    return net

ctx = mx.gpu() # 训练的时候为了简化计算，使用了单 GPU
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

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

# 训练模型
net = build_model()

epochs = 100
batch_size = 128
lr_sch = mx.lr_scheduler.FactorScheduler(step=1500, factor=0.5)
trainer = gluon.Trainer(net.collect_params(), 'adam',
                        {'learning_rate': 1e-3, 'lr_scheduler': lr_sch})

for epoch in range(epochs):
    train_loss = 0.
    train_acc = 0.
    steps = len(data_iter_train)
    for data, label in data_iter_train:
        # data = gluon.utils.split_and_load(data, ctx)
        # label = gluon.utils.split_and_load(label, ctx)
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)

        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)

    print("Epoch %d. loss: %.4f, acc: %.2f%%" % (epoch+1, train_loss/steps, train_acc/steps*100))

# 计算在训练集上的loss和准确率
evaluate(net, data_iter_train)
# 读取之前导出的测试集特征
features_test = [nd.load('features_test_%s.nd' % model_name)[0] for model_name in model_names]
features_test = nd.concat(*features_test, dim=1)
# 预测并输出到csv文件
output = nd.softmax(net(features_test.as_in_context(ctx))).asnumpy()
df_pred = pd.read_csv('sample_submission.csv')

for i, c in enumerate(df_pred.columns[1:]):
    df_pred[c] = output[:,i]

df_pred.to_csv('pred.csv', index=None)

# 和之前的提交进行对比，确认没有错位
# zip(np.argmax(pd.read_csv('pred_0.28.csv').values[:,1:], axis=-1), np.argmax(df_pred.values[:,1:], axis=-1))[:10]

# 压缩为zip文件
# !rm pred.zip
# !zip pred.zip pred.csv