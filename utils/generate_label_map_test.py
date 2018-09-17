from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap
from google.protobuf import text_format

label_map_path = './imagenet_label_map.pbtxt'

x = StringIntLabelMap()
fid = open(label_map_path, 'r')
text_format.Merge(fid.read(), x)
fid.close()