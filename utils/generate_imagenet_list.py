import os
import numpy as np
import glob


def _generate_train_list(data_path, list_name):

    sub_dirs = os.listdir(data_path)
    num_sub_dirs = len(sub_dirs)
    lines = list()
    for id, sub_dir in enumerate(sub_dirs):
        print("Processing %d/%d" % (id, num_sub_dirs))
        imgs = os.listdir(os.path.join(data_path, sub_dir))
        imgs = [img.strip('.xml') for img in imgs if img.endswith('.xml')]
        for img in imgs:
            lines += ['%s\n' % (os.path.join(sub_dir, img))]

    np.random.shuffle(lines)
    fid = open(list_name, 'w+')
    for line in lines:
        fid.write(line)
    fid.close()

def _generate_val_list(data_path, list_name):
    fid = open(list_name, 'w+')
    imgs = os.listdir(data_path)
    imgs = [img.strip('.xml') for img in imgs if img.endswith('.xml')]
    val_folder = data_path.rsplit('/')[-1]
    lines = list()
    for img in imgs:
        lines += ['%s\n' % (os.path.join(val_folder, img))]

    np.random.shuffle(lines)
    fid = open(list_name, 'w+')
    for line in lines:
        fid.write(line)
    fid.close()


train_data_path = '/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_DET/BOX/ILSVRC2014_DET_bbox_train/'
train_list_path = '../data/train_image_list'
val_data_path = '/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_DET/BOX/ILSVRC2013_DET_bbox_val/'
val_list_path = '../data/val_image_list'

_generate_train_list(train_data_path, train_list_path)
_generate_val_list(val_data_path, val_list_path)



