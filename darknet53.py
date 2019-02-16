import tensorflow as tf
from layers import Layers

class Darknet53(Layers):

    def __init__(self, is_training=False):
        self.is_training = is_training
        
    def dark_graph(self, input):
        '''
        Graph of the Darknet-53 model.
        
        Args:
            input -- 4-D tensor of shape [batch, in_height, in_width, in_channels].

        Return:
            route_1 -- 4-D tensor of the darknet of shape [batch, in_height/32, in_width/32, 1024].
            route_2 -- 4-D tensor of the darknet of shape [batch, in_height/16, in_width/16, 512].
            route_3 -- 4-D tensor of the darknet of shape [batch, in_height/8, in_width/8, 256].

        '''
        #---------------------------------------------------------------------------------
        input = self.conv2d_bn(input=input,
                        num_kernels=32)
        #---------------------------------------------------------------------------------
        input = self.conv2d_bn(input=input,
                        num_kernels=64,
                        strides=2)
        input = self._residual_block(input=input,
                                num_kernels=32)
        #---------------------------------------------------------------------------------
        input = self.conv2d_bn(input=input,
                        num_kernels=128,
                        strides=2)
        for _ in range(2):
            input = self._residual_block(input=input,
                                    num_kernels=64)
        #---------------------------------------------------------------------------------
        input = self.conv2d_bn(input=input,
                        num_kernels=256,
                        strides=2)
        for _ in range(8):
            input = self._residual_block(input=input,
                                    num_kernels=128)
        route_3 = input
        #---------------------------------------------------------------------------------
        input = self.conv2d_bn(input=input,
                        num_kernels=512,
                        strides=2)
        for _ in range(8):
            input = self._residual_block(input=input,
                                    num_kernels=256)
        route_2 = input
        #---------------------------------------------------------------------------------
        input =self.conv2d_bn(input=input,
                        num_kernels=1024,
                        strides=2)
        for _ in range(4):
            input = self._residual_block(input=input,
                                    num_kernels=512)
        
        return input, route_2, route_3

    def _residual_block(self, input, num_kernels):
        shortcut = input
        input = self.conv2d_bn(input=input,
                                num_kernels=num_kernels,
                                kernel_size=1)
        input = self.conv2d_bn(input=input,
                                num_kernels=num_kernels * 2)
        input += shortcut
        return input