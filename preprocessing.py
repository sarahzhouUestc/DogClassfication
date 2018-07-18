# encoding=utf-8
"""
构建 for_train 和 for_test 的文件夹结构
"""
import shutil
import os
import pandas as pd
df = pd.read_csv('labels.csv')
path = 'for_train'
if os.path.exists(path):
    shutil.rmtree(path)

for i, (fname, breed) in df.iterrows():
    path2 = '%s/%s' % (path, breed)
    if not os.path.exists(path2):
        os.makedirs(path2)
    os.symlink('../../train/%s.jpg' % fname, '%s/%s.jpg' % (path2, fname))

df = pd.read_csv('sample_submission.csv')
path = 'for_test'
breed = '0'

if os.path.exists(path):
    shutil.rmtree(path)

for fname in df['id']:
    path2 = '%s/%s' % (path, breed)
    if not os.path.exists(path2):
        os.makedirs(path2)
    os.symlink('../../test/%s.jpg' % fname, '%s/%s.jpg' % (path2, fname))

