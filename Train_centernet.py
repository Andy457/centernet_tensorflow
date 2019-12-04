from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils import tfrecord_voc_utils as voc_utils
import tensorflow as tf
import numpy as np
import CenterNet as net
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
lr = 0.001
batch_size = 2
buffer_size = 16
epochs = 16
reduce_lr_epoch = []
config = {
    'mode': 'train',                                       # 'train', 'test'
    'input_size': 416,
    'data_format': 'channels_last',                        # 'channels_last' 'channels_first'
    'num_classes': 9,
    'weight_decay': 1e-4,
    'keep_prob': 0.5,                                      # not used
    'batch_size': batch_size,

    'score_threshold': 0.1,                                 
    'top_k_results_output': 100,
}

image_augmentor_config = {
    'data_format': 'channels_last',
    'output_shape': [416, 416],
    'zoom_size': [448, 448],
    'crop_method': 'random',
    'flip_prob': [0., 0.5],
    'fill_mode': 'BILINEAR',
    'keep_aspect_ratios': False,
    'constant_values': 0.,
    'color_jitter_prob': 0.5,
    'rotate': [0.5, -5., -5.],
    'pad_truth_to': 60,
}

data_address = 'D:/temp_models/Object-Detection-API-Tensorflow-master/VOC2007/ImageSets/'
data = os.listdir(data_address)
data = [os.path.join(data_address, name) for name in data]

train_gen = voc_utils.get_generator(data, batch_size, buffer_size, image_augmentor_config)
trainset_provider = \
    {
    'data_shape': [416, 416, 3],
    'num_train': 8432,
    'num_val': 0,                                         # not used
    'train_generator': train_gen,
    'val_generator': None                                 # not used
    }
centernet = net.CenterNet(config, trainset_provider)
for i in range(epochs):
    print('-'*25, 'epoch', i, '-'*25)
    if i in reduce_lr_epoch:
        lr = lr/10.
        print('reduce lr, lr=', lr, 'now')
    mean_loss = centernet.train_one_epoch(lr)
    print('>> mean loss', mean_loss)
    centernet.save_weight('latest', './centernet/test')            # 'latest', 'best