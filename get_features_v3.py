# encoding=utf-8
import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon.data import vision
from mxnet.gluon.model_zoo import vision as models
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import h5py
import os

import matplotlib.pyplot as plt
ctx = [mx.gpu(i) for i in range(4)]

# 载入训练集
df = pd.read_csv('labels.csv')
synset = sorted(set(df['breed']))
n = len(df)

X_224 = nd.zeros((n, 3, 224, 224))
X_299 = nd.zeros((n, 3, 299, 299))
y = nd.zeros((n,))

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

for i, (fname, breed) in tqdm(df.iterrows(), total=n):
    img = cv2.imread('train/%s.jpg' % fname)
    img_224 = ((cv2.resize(img, (224, 224))[:, :, ::-1] / 255.0 - mean) / std).transpose((2, 0, 1))
    img_299 = ((cv2.resize(img, (299, 299))[:, :, ::-1] / 255.0 - mean) / std).transpose((2, 0, 1))

    X_224[i] = nd.array(img_224)
    X_299[i] = nd.array(img_299)

    y[i] = synset.index(breed)

    nd.waitall()
trainnd_path = os.path.join('~','PengXiao/breed/train.nd')
labelnd_path = os.path.join('~','PengXiao/breed/labels.nd')
if not os.path.exists(trainnd_path):
    nd.save('train.nd', [X_224, X_299, y])
if not os.path.exists(labelnd_path):
    nd.save('labels.nd', y)

# 载入测试集
df_test = pd.read_csv('sample_submission.csv')
n_test = len(df_test)

X_224_test = nd.zeros((n_test, 3, 224, 224))
X_299_test = nd.zeros((n_test, 3, 299, 299))

for i, fname in tqdm(enumerate(df_test['id']), total=n_test):
    img = cv2.imread('test/%s.jpg' % fname)
    img_224 = ((cv2.resize(img, (224, 224))[:, :, ::-1] / 255.0 - mean) / std).transpose((2, 0, 1))
    img_299 = ((cv2.resize(img, (299, 299))[:, :, ::-1] / 255.0 - mean) / std).transpose((2, 0, 1))

    X_224_test[i] = nd.array(img_224)
    X_299_test[i] = nd.array(img_299)

    nd.waitall()

testnd_path = os.path.join('~','PengXiao/breed/test.nd')
if not os.path.exists(testnd_path):
    nd.save('test.nd', [X_224_test, X_299_test])

# 检查点
X_224, X_299, y = nd.load('train.nd')
X_224_test, X_299_test = nd.load('test.nd')

# 导出特征
def save_features(model_name, data_train_iter, data_test_iter, ignore=False):
    # 文件已存在
    if os.path.exists('features_train_%s.nd' % model_name) and ignore:
        if os.path.exists('features_test_%s.nd' % model_name):
            return

    net = models.get_model(model_name, pretrained=True, ctx=ctx)

    for prefix, data_iter in zip(['train', 'test'], [data_train_iter, data_test_iter]):
        features = []
        for data in tqdm(data_iter):
            # 并行预测数据
            for data_slice in gluon.utils.split_and_load(data, ctx, even_split=False):
                feature = net.features(data_slice)
                if 'squeezenet' in model_name:
                    feature = gluon.nn.GlobalAvgPool2D()(feature)
                feature = gluon.nn.Flatten()(feature)
                features.append(feature.as_in_context(mx.cpu()))
            nd.waitall()

        features = nd.concat(*features, dim=0)
        nd.save('features_%s_%s.nd' % (prefix, model_name), features)

batch_size = 64

data_iter_224 = gluon.data.DataLoader(gluon.data.ArrayDataset(X_224), batch_size=batch_size)
data_iter_299 = gluon.data.DataLoader(gluon.data.ArrayDataset(X_299), batch_size=batch_size)

data_test_iter_224 = gluon.data.DataLoader(gluon.data.ArrayDataset(X_224_test), batch_size=batch_size)
data_test_iter_299 = gluon.data.DataLoader(gluon.data.ArrayDataset(X_299_test), batch_size=batch_size)
from mxnet.gluon.model_zoo.model_store import _model_sha1

for model in sorted(_model_sha1.keys()):
    print(model)
    if model == 'inceptionv3':
        save_features(model, data_iter_299, data_test_iter_299)
    else:
        save_features(model, data_iter_224, data_test_iter_224)

from mxnet.gluon.model_zoo.model_store import _model_sha1

for model in sorted(_model_sha1.keys()):
    print(model)
    if model == 'inceptionv3':
        save_features(model, data_iter_299, data_test_iter_299, ignore=True)
    else:
        save_features(model, data_iter_224, data_test_iter_224, ignore=True)

