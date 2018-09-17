import sys
sys.path.append('external')
from object_detection.core import target_assigner
from object_detection.builders import box_coder_builder, matcher_builder, box_predictor_builder
from object_detection.builders import anchor_generator_builder, losses_builder
from object_detection.builders import image_resizer_builder, post_processing_builder
from object_detection.builders import region_similarity_calculator_builder as sim_calc_builder
from object_detection.protos import model_pb2
from object_detection.core import box_list
from google.protobuf import text_format
import tensorflow as tf


# input
model = model_pb2.DetectionModel()
f = open('model.config', 'r')
text_format.Merge(f.read(), model)
f.close()
num_classes = 20
groundtruth_class = tf.get_variable('groundtruth_class', shape=[24, 5, 20])
groundtruth_box = tf.get_variable('groundtruth_box', shape=[24, 5, 4])

groundtruth_classes_with_background_list = [ tf.pad(one_hot_encoding, [[0, 0], [1, 0]], mode='CONSTANT')
                            for one_hot_encoding in tf.unstack(groundtruth_class)]
groundtruth_boxlists = [ box_list.BoxList(boxes)
                        for boxes in tf.unstack(groundtruth_box)]



# construct models
box_coder = box_coder_builder.build(model.ssd.box_coder)
matcher = matcher_builder.build(model.ssd.matcher)
region_similarity_calculator = sim_calc_builder.build(model.ssd.similarity_calculator)
anchor_generator = anchor_generator_builder.build(model.ssd.anchor_generator)
(classification_loss, localization_loss, classification_weight,
 localization_weight, hard_example_miner) = losses_builder.build(model.ssd.loss)
image_resizer_fn = image_resizer_builder.build(model.ssd.image_resizer)
non_max_suppression_fn, score_conversion_fn = post_processing_builder.build(model.ssd.post_processing)
(classification_loss, localization_loss,
 classification_weight,localization_weight,hard_example_miner) = losses_builder.build(model.ssd.loss)
normalize_loss_by_num_matches = model.ssd.normalize_loss_by_num_matches
matcher = matcher_builder.build(model.ssd.matcher)
unmatched_cls_target = tf.constant([1] + num_classes * [0], tf.float32)
_target_assigner = target_assigner.TargetAssigner(
        region_similarity_calculator,
        matcher,
        box_coder,
        positive_class_weight=1.0,
        negative_class_weight=1.0,
        unmatched_cls_target=unmatched_cls_target)

anchors = anchor_generator.generate([(i,i) for i in range(9,3,-1)])
pass

a = target_assigner.batch_assign_targets(_target_assigner, anchors, groundtruth_boxlists,
        groundtruth_classes_with_background_list)
1
2
pass