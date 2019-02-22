import tensorflow as tf
from darknet53 import Darknet53

class Yolov3(Darknet53):
    
    _ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]
    objectness = 1
    coordinates=4
    num_classes = 80
    num_predictions = objectness + coordinates + num_classes            

    def __init__(self, input, is_training=False):
        self.is_training = is_training
        self.sess = tf.Session()
        self.input = input
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

    def _transform_predictions(self, predictions, anchor_list):
        num_anchors = len(anchor_list)
        output_shape = predictions.get_shape().as_list() # if output_shape=(m, 13, 13, 255)
        grid_sz = output_shape[1] # grid_sz = 13
        grid_dim = grid_sz * grid_sz # grid_dim = 169
        stride = self.input_size // grid_sz
        
        predictions = tf.reshape(predictions, [-1, grid_dim*num_anchors, self.num_predictions]) # predictions = (m, 507, 85)
        
        anchors_xy, anchors_hw, confidence, classes = tf.split(predictions, [2, 2, 1, self.num_classes], axis=-1) # split along the last dimension

        anchors_xy = tf.sigmoid(anchors_xy)
        confidence = tf.sigmoid(confidence)
        classes = tf.sigmoid(classes)
        
        grid_sz_range = tf.range(grid_sz, dtype=tf.float32)
        grid_x, grid_y = tf.meshgrid(grid_sz_range, grid_sz_range)
        grid_x = tf.reshape(grid_x, [-1, 1])
        grid_y = tf.reshape(grid_y, [-1, 1])
        grid_offset = tf.concat([grid_x, grid_y], axis=-1)
        grid_offset = tf.tile(grid_offset, [1, num_anchors])
        grid_offset = tf.reshape(grid_offset, [1, -1, 2])
        
        anchors_xy = anchors_xy + grid_offset
        anchors_xy = anchors_xy * stride
        
        anchors = [(a[0] / stride, a[1] / stride) for a in anchor_list]
        anchors = tf.tile(anchors, [grid_dim, 1])
        
        anchors_hw = anchors * tf.exp(anchors_hw) * stride
        
        return tf.concat([anchors_xy, anchors_hw, confidence, classes], axis=-1) # concat along the last dimension

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
        
        predictions = self.conv2d(input=input,
                            num_kernels=output_depth,
                            with_bias=True)

        predictions = self._transform_predictions(predictions, anchor_list)
        
        return predictions, route
        