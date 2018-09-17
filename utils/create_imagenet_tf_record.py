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

import hashlib
import io
import logging
import os
import numpy as np
from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util



flags = tf.app.flags
flags.DEFINE_string('data_dir', '/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_DET/image/ILSVRC2014_DET_train',
                    'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('annotations_dir', '/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_DET/BOX/ILSVRC2014_DET_bbox_train/',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', '/media/2TB/Research/DataSet/ILSVRC2014/train.record',
                    'Path to output TFRecord')
flags.DEFINE_string('data_list_path', 'data/train_image_list',
                    'Path to image list')
flags.DEFINE_string('label_map_path', 'data/imagenet_label_map.pbtxt',
                    'Path to label map proto')
# flags.DEFINE_string('log_path', 'log', 'path to log file')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS

# logging.basicConfig(filename=FLAGS.log_path, level=logging.INFO)


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  # Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.join(data['folder'], data['filename']+'.JPEG')
  #img_path = os.path.join(data['filename'] + '.JPEG')
  full_path = os.path.join(dataset_directory, img_path)
  with tf.gfile.GFile(full_path, 'rb') as fid:
      encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
      print("Image format is %s,not JPEG\n"%(image.format))
      return None
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])
  max_size = max(width,height)
  if max_size > 500:
      scale = 500.0 / max_size
      image = image.resize(np.int32([width * scale, height * scale]))
      image.save('./tmp/tmp.JPEG')
      with tf.gfile.GFile('./tmp/tmp.JPEG', 'rb') as fid:
          encoded_jpg = fid.read()


  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  for obj in data['object']:
    xmin.append(float(obj['bndbox']['xmin']) / width)
    ymin.append(float(obj['bndbox']['ymin']) / height)
    xmax.append(float(obj['bndbox']['xmax']) / width)
    ymax.append(float(obj['bndbox']['ymax']) / height)
    classes_text.append(obj['name'].encode('utf8'))
    classes.append(label_map_dict[obj['name']])

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return example


def main(_):
    data_dir = FLAGS.data_dir

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    # logging.info('Reading from Imagenet dataset.')
    examples_list = dataset_util.read_examples_list(FLAGS.data_list_path)
    for idx, example in enumerate(examples_list):
        if idx % 100 == 0:
            print('On image %d of %d'%(idx, len(examples_list)))
            # logging.info('On image %d of %d', idx, len(examples_list))
        path = os.path.join(FLAGS.annotations_dir, example + '.xml')
        with tf.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        if not data.has_key('object'):
            continue
        tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
                                      FLAGS.ignore_difficult_instances)
        if tf_example is not None:
            writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
  tf.app.run()
