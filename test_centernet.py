from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils import tfrecord_voc_utils as voc_utils
import tensorflow as tf
import numpy as np
import CenterNet as net
import os
import time
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io, transform
from utils.voc_classname_encoder import classname_to_ids


lr = 0.001
batch_size = 4
buffer_size = 16    # the meaning?
epochs = 16
reduce_lr_epoch = []
config = {
    'mode': 'test',                                       # 'train', 'test'
    'input_size': 416,
    'data_format': 'channels_last',                        # 'channels_last' 'channels_first'
    'num_classes': 8,
    'weight_decay': 1e-4,
    'keep_prob': 0.5,                                      # not used
    'batch_size': batch_size,

    'score_threshold': 0.25,
    'top_k_results_output': 10,
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
centernet.load_weight('./centernet/test-99500')

save_addesss = 'E:/centernet/'
image_address = 'E:/changdain_ok/'
imagelist = os.listdir(image_address)
for image in imagelist:
    total_address = image_address + image
    img = io.imread(total_address)
    img = transform.resize(img, [416, 416])
    img1 = np.expand_dims(img, 0)
    tic = time.time()
    result = centernet.test_one_image(img1)
    toc = time.time()
    print('time:', toc-tic)
    id_to_clasname = {k:v for (v,k) in classname_to_ids.items()}
    scores = result[0]
    bbox = result[1]
    class_id = result[2]
    classes = [id_to_clasname[key] for key in class_id]

    plt.figure(1)
    plt.imshow(np.squeeze(img))
    axis = plt.gca()
    for i in range(len(scores)):
        rect = patches.Rectangle((bbox[i][1],bbox[i][0]), bbox[i][3]-bbox[i][1],bbox[i][2]-bbox[i][0],linewidth=2,edgecolor='b',facecolor='none')
        axis.add_patch(rect)
        plt.text(bbox[i][1],bbox[i][0], id_to_clasname[class_id[i]]+str(' ')+str(scores[i]), color='red', fontsize=12)
    plt.savefig(save_addesss+image)
    # plt.show()
    plt.close()