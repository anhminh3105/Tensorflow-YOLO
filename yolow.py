import tensorflow as tf
from yolov3 import *

class Yolow(Yolov3):
    
    def __init__(self, input_shape=[None, 416, 416, 3], weight_path=None, is_training=False):
        self.is_training = is_training
        self.sess = tf.Session()
        self.inp = tf.placeholder(tf.float32, input_shape, 'input')
        with tf.variable_scope('detections'):
            self.outp = self.graph(self.inp) 
        self.loader = Weight_loader(tf.global_variables('detections'), weight_path)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.loader.load_now())

    def predict(self, input_list, confidence_theshold=.6, iou_threshold=.5):
        feed_dict = {self.inp: input_list}
        batch_detections = self.sess.run(self.outp, feed_dict)
        return predict(batch_detections, confidence_theshold, iou_threshold)
