import os
try:
    import cPickle as pickle
except:
    import pickle

from im2col import filter_to_2d, data_to_2d
import csv

FILE_PATH = '/opt/storage/data/resnet/cifar-10/resnet-l20-b128/tensors.pickle'

with open(FILE_PATH, 'r') as f:
    data = pickle.load(f)

# NHWC
images4d = data[0]
# RSCK
weights4d = data[1]

print (images4d.shape)
print (weights4d.shape)

# NCHW
images4d = images4d.transpose((0, 3, 1, 2))
# KCRS
weights4d = weights4d.transpose((3, 2, 0, 1))

print (images4d.shape)
print (weights4d.shape)

weights2d = filter_to_2d(weights4d)
images2d = data_to_2d(images4d, weights4d, 1, 1, 2, 2)

print (images2d.shape)
print (weights2d.shape)

IMAGES_CSV_NAME = 'images_%dx%d.csv' % (images2d.shape[0], images2d.shape[1])
WEIGHTS_CSV_NAME = 'weights_%dx%d.csv' % (weights2d.shape[0], weights2d.shape[1])

with open(IMAGES_CSV_NAME, 'w') as f:
    csvWriter = csv.writer(f)
    listData = []
    for i in range(images2d.shape[0]):
        for j in range(images2d.shape[1]):
            listData.append(images2d[i][j])
    csvWriter.writerow(listData)

with open(WEIGHTS_CSV_NAME, 'w') as f:
    csvWriter = csv.writer(f)
    listData = []
    for i in range(weights2d.shape[0]):
        for j in range(weights2d.shape[1]):
            listData.append(weights2d[i][j])
    csvWriter.writerow(listData)
