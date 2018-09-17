nohup python -u utils/create_imagenet_tf_record.py \
  --data_dir='/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_DET/image/ILSVRC2013_DET_val' \
  --annotations_dir='/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_DET/BOX/ILSVRC2013_DET_bbox_val' \
  --output_path='/media/2TB/Research/DataSet/ILSVRC2014/val_maxsz500.record' \
  --data_list_path='data/val_image_list' \
  2>&1 1>log_val &

