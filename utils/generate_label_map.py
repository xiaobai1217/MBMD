import scipy as sp
import scipy.io
imagenet_meta_data_path = '/home/lijun/Research/DataSet/ILSVRC2014/ILSVRC2014_devkit/ILSVRC2014_devkit/data/meta_det.mat'
label_map_path = './imagenet_label_map.pbtxt'
meta_data = sp.io.loadmat(imagenet_meta_data_path, struct_as_record=False)
synsets = meta_data['synsets']

fid = open(label_map_path, 'w')

for i in range(200):
    id = synsets[0,i].ILSVRC2013_DET_ID[0,0]
    category = synsets[0,i].WNID[0].encode('ascii')
    fid.write('item{\n')
    fid.write('  id:%d\n'%id)
    fid.write('  name: \'%s\'\n'%category)
    fid.write('}\n')
fid.close()