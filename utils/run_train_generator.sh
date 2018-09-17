nohup python -u utils/create_imagenet_tf_record.py \
  --data_dir='/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_DET/image/ILSVRC2014_DET_train' \
  --annotations_dir='/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_DET/BOX/ILSVRC2014_DET_bbox_train/' \
  --output_path='/media/2TB/Research/DataSet/ILSVRC2014/train_maxsz500.record' \
  --data_list_path='data/train_image_list' \
  2>&1 1>log_train &
