# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    ./create_pascal_tf_record --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from lxml import etree
import os
import PIL.Image



flags = tf.app.flags
flags.DEFINE_string('data_root', '/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_DET/',
                    'Root directory to imagenet detection data set')
flags.DEFINE_string('image_path', 'image/ILSVRC2014_DET_train/',
                    '(Relative) path to image directory.')
flags.DEFINE_string('ann_path', 'BOX/ILSVRC2014_DET_bbox_train/',
                    '(Relative) path to annotation directory')
flags.DEFINE_string('data_list_path', '/media/2TB/Research/Code/memory_augmented_tracker/data/train_image_list',
                    'Path to image list.')
flags.DEFINE_string('res_path', '/media/2TB/Research/DataSet/ILSVRC2014/train_seq.record',
                    'Path to write tfrecord.')

FLAGS = flags.FLAGS



def recursive_parse_xml(xml):
    if len(xml) == 0:
        return xml.text
    result = dict()
    for child in xml:
        child_res = recursive_parse_xml(child)
        if child.tag == 'object':
            if child.tag in result:
                result[child.tag].append(child_res)
            else:
                result[child.tag] = list([child_res])
        else:
            result[child.tag] = child_res
    return result
def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))



def main(_):
    data_root = FLAGS.data_root
    image_path = FLAGS.image_path
    ann_path = FLAGS.ann_path
    image_path = os.path.join(data_root, image_path)
    ann_path = os.path.join(data_root, ann_path)
    data_list = FLAGS.data_list_path
    res_path = FLAGS.res_path

    with open(data_list, 'r') as fid:
        data_list = fid.readlines()
    data_list = [data.strip('\n') for data in data_list]

    num_imgs = len(data_list)
    writer = tf.python_io.TFRecordWriter(res_path)
    for ind, sample in enumerate(data_list):
        if ind % 100 == 0:
            print("On Image %d/%d\n" % (ind, num_imgs))
        with open(os.path.join(ann_path, sample + '.xml')) as fid:
            xml = etree.fromstring(fid.read())
        xml = recursive_parse_xml(xml)
        img = PIL.Image.open(os.path.join(image_path, sample + '.JPEG'))
        img_array = np.array(img)
        if img.format is not 'JPEG' or img_array.ndim == 3 and img_array.shape[2] !=3 \
                or 'object' not in xml.keys():
            continue
        width = int(xml['size']['width'])
        height = int(xml['size']['height'])
        for obj in xml['object']:
            xmin, xmax, ymin, ymax = (float(obj['bndbox']['xmin'])/width, float(obj['bndbox']['xmax'])/width,
                                      float(obj['bndbox']['ymin'])/height, float(obj['bndbox']['ymax'])/height)
            if xmin>=xmax or ymin>=ymax or xmin<0 or xmax>1 or ymin<0 or ymax>1:
                continue

            example = tf.train.Example(features=tf.train.Features(feature={
                'bndbox/xmin': _float_list_feature([xmin]),
                'bndbox/xmax': _float_list_feature([xmax]),
                'bndbox/ymin': _float_list_feature([ymin]),
                'bndbox/ymax': _float_list_feature([ymax]),
                'image_name': _bytes_list_feature([sample])}))
            writer.write(example.SerializeToString())
    writer.close()
    print('Create TFRecord Success!')


if __name__ == '__main__':
  tf.app.run()
