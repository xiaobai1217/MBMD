
from external.object_detection.anchor_generators.multiple_grid_anchor_generator import create_ssd_anchors

ssd_anchor_generator = create_ssd_anchors(num_layers=3, min_scale=0.2, max_scale=0.95,
                                      aspect_ratios=[1.0, 2.0, 0.5, 3.0, 0.33329999446868896])
anchors = ssd_anchor_generator.generate([(20, 20),(10, 10),(4,4)])


