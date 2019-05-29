import tensorflow as tf
from darknet53 import Darknet53
from utils import *

class Yolov3(Darknet53):
    
    _ANCHORS = anchors_for_yolov3()
    objectness = 1
    coordinates=4
    num_classes = 80
    num_predictions = objectness + coordinates + num_classes            

    def __init__(self, input, is_training=False):
        super().__init__(input=input, is_training=is_training)
        self.sess = tf.Session()
        self.input_size = self.input.get_shape().as_list()[1]

    def graph(self):
        with tf.variable_scope('darknet53'):
            route_1, route_2, route_3 = self.dark_graph()
        with tf.variable_scope('yolov3'):
            predictions_1, route = self._detection_block(route_1, 512, self._ANCHORS[6:])
            route = self.conv2d_bn(input=route,
                            num_kernels=256,
                            kernel_size=1)
            route = self._upsample(route)
            route_2 = tf.concat([route, route_2], axis=-1)
            predictions_2, route = self._detection_block(route_2, 256, self._ANCHORS[3:6])
            route = self.conv2d_bn(input=route,
                            num_kernels=128,
                            kernel_size=1)
            route = self._upsample(route)
            route_3 = tf.concat([route, route_3], axis=-1)
            predictions_3, route = self._detection_block(route_3, 128, self._ANCHORS[:3])
            
        return tf.concat([predictions_1, predictions_2, predictions_3], axis=1, name='output')

    def _upsample(self, input, stride=2):
        grid_shape = input.get_shape().as_list()
        grid_sz = grid_shape[1]
        output_dim = grid_sz*stride
        return tf.image.resize_nearest_neighbor(images=input,
                                                size=(output_dim, output_dim),
                                                name='upsampled_' + str(output_dim))

    def _detection_block(self, input, num_kernels, anchor_list):
        num_anchors = len(anchor_list)
        output_depth = num_anchors * self.num_predictions
        for i in range(3):
            input = self.conv2d_bn(input=input,
                            num_kernels=num_kernels,
                            kernel_size=1)
            if i == 2:
                route = input

            input = self.conv2d_bn(input=input,
                            num_kernels=num_kernels*2)
        
        detections = self.detection(input, output_depth, anchor_list, self.input_size)
        
        return detections, route
        