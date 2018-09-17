import tensorflow as tf
import numpy as np
from lxml import etree
import os
import PIL.Image as Image
from collections import OrderedDict
from utils.create_imagenet_seq_tf_record import recursive_parse_xml, _bytes_list_feature, _float_list_feature, _int64_list_feature



image_root = '/media/2TB/Research/DataSet/ILSVRC2015/Data/VID/train/'
ann_root = '/media/2TB/Research/DataSet/ILSVRC2015/Annotations/VID/train/'
res_path = '/media/2TB/Research/DataSet/ILSVRC2015/TFRecord/train.record'


def main(_):
    folders = os.listdir(image_root)
    folders.sort()
    snippet_list = list()
    for folder in folders:
        snippets = os.listdir(os.path.join(image_root, folder))
        snippets.sort()
        snippet_list.extend([os.path.join(folder,snippet) for snippet in snippets])
    np.random.shuffle(snippet_list)

    writer = tf.python_io.TFRecordWriter(res_path)
    num_snippets = len(snippet_list)

    for sid, snippet in enumerate(snippet_list):
        if sid%100 == 0:
            print("On snippet: %d / %d"%(sid, num_snippets))
        frames = os.listdir(os.path.join(image_root, snippet))
        frames.sort()
        object_dict = OrderedDict()
        for frame in frames:
            img = Image.open(os.path.join(image_root, snippet, frame))
            img_array = np.array(img)
            height, width = img_array.shape[0:2]
            if img.format != 'JPEG' or img_array.ndim == 3 and img_array.shape[2] > 3:
                continue

            xml_file = os.path.join(ann_root, snippet, frame.rstrip('JPEG') + 'xml')
            with open(xml_file) as fid:
                xml = etree.fromstring(fid.read())
            xml = recursive_parse_xml(xml)
            if 'object' not in xml:
                continue
            for obj in xml['object']:
                xmin, xmax, ymin, ymax = (
                    float(obj['bndbox']['xmin']) / width, float(obj['bndbox']['xmax']) / width,
                    float(obj['bndbox']['ymin']) / height, float(obj['bndbox']['ymax']) / height)

                if xmin >= xmax or ymin >= ymax or xmin < 0 or xmax > 1 or ymin < 0 or ymax > 1:
                    continue

                if obj['trackid'] not in object_dict.keys():
                    object_dict[obj['trackid']] = dict({'frame': list(),
                                                        'xmax': list(),
                                                        'xmin': list(),
                                                        'ymax': list(),
                                                        'ymin': list()})
                object_dict[obj['trackid']]['frame'].append(frame.rstrip('.JPEG'))
                object_dict[obj['trackid']]['xmax'].append(xmax)
                object_dict[obj['trackid']]['xmin'].append(xmin)
                object_dict[obj['trackid']]['ymax'].append(ymax)
                object_dict[obj['trackid']]['ymin'].append(ymin)

        for obj in object_dict.values():
            example = tf.train.Example(features=tf.train.Features(feature={
                'bndbox/xmin': _float_list_feature(obj['xmin']),
                'bndbox/xmax': _float_list_feature(obj['xmax']),
                'bndbox/ymin': _float_list_feature(obj['ymin']),
                'bndbox/ymax': _float_list_feature(obj['ymax']),
                'image_name': _bytes_list_feature(obj['frame']),
                'folder': _bytes_list_feature([snippet])}))
            writer.write(example.SerializeToString())
    writer.close()
    print('Create TFRecord Success!')


if __name__ == '__main__':
    tf.app.run()