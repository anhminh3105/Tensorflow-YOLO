import tensorflow as tf

class Layers(object):
    
    def __init__(self, is_training=False):
        '''
        Args:
            is_training -- Whether to return the output in training mode (normalized with statistics of the current batch) or in inference mode (normalized with moving statistics).
        '''
        self.is_training = is_training

    def conv2d_bn(self, input, num_kernels, kernel_size=3, strides=1):
        '''
        A buiding block of DarkNet53, a 2D convolution layer followed by a batch normalisation layer.
        
        Args:
            input -- 4-D tensor of shape [batch, in_height, in_width, in_channels].
            num_kernels -- the number of kernels.
            kernel_size -- the kernel size.
            strides -- the stride of the sliding for kernels. 
            is_training -- Whether to return the output in training mode (normalized with statistics of the current batch) or in inference mode (normalized with moving statistics).
        Returns:
            leaky ReLU applied outputs.
        '''
        z = self.conv2d(input=input,
                    num_kernels=num_kernels,
                    kernel_size=kernel_size,
                    strides=strides)
        z_tilder = tf.layers.batch_normalization(inputs=z,
                                                momentum=.9,
                                                epsilon=1e-05,
                                                training=self.is_training)    
        return tf.nn.leaky_relu(z_tilder, alpha=.1)

    def conv2d(self, input, num_kernels, kernel_size=1, strides=1, with_bias=False):
        '''
        An implementation of a 2D convolution layer.

        Args:
            input -- 4-D tensor of shape [batch, in_height, in_width, in_channels].
            num_kernels -- the number of kernels.
            kernel_size -- the kernel size.
            strides -- the stride of the sliding for kernels. 
            with_bias -- Whether to use biases in the layer.
        Returns:
            outputs.
        '''
        with tf.variable_scope(name_or_scope=None, default_name='conv2d'):
            if kernel_size > 1:
                paddings = tf.constant([[0,0], [1,1], [1,1], [0,0]])
                input = tf.pad(input, paddings)
            W = tf.get_variable(name='weights',
                                shape=[kernel_size, kernel_size, input.shape[-1], num_kernels],
                                dtype=tf.float32)

            z = tf.nn.conv2d(input=input,
                            filter=W,
                            strides=[1, strides, strides, 1],
                            padding='VALID')
            if with_bias:
                # Darknet53 uses tied biases.
                bias = tf.get_variable(name='biases',
                                    shape=num_kernels,
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer)
                z += bias

            return z