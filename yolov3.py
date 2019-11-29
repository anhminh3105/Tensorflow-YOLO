import tensorflow as tf
import numpy as np

from darknet53 import Darknet53
from utils import *

class Yolov3(Darknet53):
    
    num_anchors = 3
    objectness = 1
    coordinates = 4

    def __init__(self, input, num_classes, input_size, anchor_file, is_training):
        super().__init__(input=input, is_training=is_training)
        self._ANCHORS = load_anchors(anchor_file)
        self.num_classes = num_classes
        self.num_predictions = self.objectness + self.coordinates + self.num_classes # 5 includes objectness(1), coordinates(4)
        self.input_size = self.input.get_shape().as_list()[1:3]

    def _detection_block(self, input, num_kernels, anchor_list):
        output_depth = self.num_anchors * self.num_predictions
        for i in range(3):
            input = self.conv2d_bn(input=input,
                                   num_kernels=num_kernels,
                                   kernel_size=1)
            if i == 2:
                route = input
            input = self.conv2d_bn(input=input,
                                   num_kernels=num_kernels*2)
        input = self.conv2d(input=input,
                            num_kernels=output_depth,
                            with_bias=True)
        input = self.region(input, anchor_list)        
        return input, route


    def region(self, input, anchor_list):
        output_shape = input.get_shape().as_list() # if output_shape=(m, 13, 13, 255)
        grid_sz = output_shape[1:3] # grid_sz = 13, 13
        grid_dim = np.prod(grid_sz) # grid_dim = 169
        strides = np.array(self.input_size) / np.array(grid_sz)
        
        input = tf.reshape(input, [-1, grid_dim*self.num_anchors, self.num_predictions]) # predictions = (m, 507, 85)
        
        anchors_xy, anchors_hw, confidence, classes = tf.split(input, [2, 2, 1, self.num_classes], axis=-1) # split along the last dimension

        anchors_xy = tf.sigmoid(anchors_xy)
        confidence = tf.sigmoid(confidence)
        classes = tf.sigmoid(classes)
        
        grid_x_range = tf.range(grid_sz[0], dtype=tf.float32)
        grid_y_range = tf.range(grid_sz[1], dtype=tf.float32)
        grid_x, grid_y = tf.meshgrid(grid_x_range, grid_y_range)
        grid_x = tf.reshape(grid_x, [-1, 1])
        grid_y = tf.reshape(grid_y, [-1, 1])
        grid_offset = tf.concat([grid_x, grid_y], axis=-1)
        grid_offset = tf.tile(grid_offset, [1, self.num_anchors])
        grid_offset = tf.reshape(grid_offset, [1, -1, 2])
        
        anchors_xy = anchors_xy + grid_offset
        anchors_xy = anchors_xy * strides
        
        anchors = [tuple(a/strides) for a in anchor_list]
        anchors = tf.tile(anchors, [grid_dim, 1])
        
        anchors_hw = anchors * tf.exp(anchors_hw) * strides
        
        return tf.concat([anchors_xy, anchors_hw, confidence, classes], axis=-1) # concat along the last dimension


    def _upsample(self, input, stride=2):
        grid_shape = input.get_shape().as_list()
        grid_sz = grid_shape[1]
        output_dim = grid_sz*stride
        return tf.image.resize_nearest_neighbor(images=input,
                                                size=(output_dim, output_dim),
                                                name='upsampled_' + str(output_dim))


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