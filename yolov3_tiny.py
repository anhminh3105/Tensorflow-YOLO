import tensorflow as tf
from yolov3 import Yolov3
from utils import *

class Yolov3Tiny(Yolov3):
    def _detection_block(self, input, num_kernels, anchor_list):
        output_depth = len(anchor_list)*self.num_predictions
        input = self.conv2d_bn(input=input,
                               num_kernels=num_kernels)
        input = self.conv2d(input=input,
                            num_kernels=output_depth,
                            with_bias=True)
        input = self.region(input, anchor_list)   
        return input


    def graph(self):
        with tf.variable_scope('yolov3_tiny'):
            input = self.conv2d_bn(input=self.input,
                                   num_kernels=16)
            for i in range(6):
                input = tf.layers.max_pooling2d(inputs=input,
                                                pool_size=2,
                                                strides=(1 if i == 5 else 2),
                                                padding='same' if i == 5 else 'valid')
                input = self.conv2d_bn(input=input,
                                       num_kernels=pow(2, 5+i))  
                if i == 3:
                    route_1 = input
            input = self.conv2d_bn(input=input,
                                   num_kernels=256,
                                   kernel_size=1)
            route_2 = input
            predictions_1 = self._detection_block(input=input,
                                                 num_kernels=512,
                                                 anchor_list=self._ANCHORS[3:6])       
            input = self.conv2d_bn(input=route_2,
                                   num_kernels=128,
                                   kernel_size=1)
            input = self._upsample(input)
            input = tf.concat(values=[input, route_1],
                              axis=-1)
            predictions_2 = self._detection_block(input=input,
                                                 num_kernels=256,
                                                 anchor_list=self._ANCHORS[:3])
        return tf.concat(values=[predictions_1, predictions_2],
                         axis=1,
                         name='output')