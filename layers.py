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

    def region(self, predictions, anchor_list, input_size):
        num_anchors = len(anchor_list)
        output_shape = predictions.get_shape().as_list() # if output_shape=(m, 13, 13, 255)
        grid_sz = output_shape[1] # grid_sz = 13
        grid_dim = grid_sz * grid_sz # grid_dim = 169
        stride = input_size // grid_sz
        
        predictions = tf.reshape(predictions, [-1, grid_dim*num_anchors, 85]) # predictions = (m, 507, 85)
        
        anchors_xy, anchors_hw, confidence, classes = tf.split(predictions, [2, 2, 1, 80], axis=-1) # split along the last dimension

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

    def detection(self, input, output_depth, anchor_list, input_size):
        detections = self.conv2d(input=input,
                            num_kernels=output_depth,
                            with_bias=True)

        return self.region(detections, anchor_list, input_size)