import tensorflow as tf
from yolov3 import *
from predict import predict
from load import Weight_loader

class Yolow(Yolov3):

    sess = tf.Session()
    
    def __init__(self, input=None, weight_path=None, is_training=False):
        self.is_training = is_training
        try:
            self.defrost()
            self.input = tf.get_default_graph().get_tensor_by_name('import/input:0')
            self.output = tf.get_default_graph().get_tensor_by_name('import/detections/output:0')
        except:
            if not input:
                input = tf.placeholder(tf.float32, [None, 416, 416, 3], 'input')
            self.input = input
            self.input_size = self.input.get_shape().as_list()[1]
            with tf.variable_scope('detections'):
                self.output = self.graph() 
            self.loader = Weight_loader(tf.global_variables('detections'), weight_path)
            # self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.loader.load_now())
            self.freeze()

    def predict(self, input_list, confidence_theshold=.6, iou_threshold=.5):
        feed_dict = {self.input: input_list}
        batch_detections = self.sess.run(self.output, feed_dict)
        return predict(batch_detections, confidence_theshold, iou_threshold)

    def freeze(self):
        graph_def = tf.graph_util.convert_variables_to_constants(sess=self.sess,
                                                                input_graph_def=tf.get_default_graph().as_graph_def(),
                                                                output_node_names=['detections/output'])
        with tf.gfile.GFile('frozen_yolow.pb', 'wb') as f:
            f.write(graph_def.SerializeToString())

    def defrost(self):
        with tf.gfile.GFile('frozen_yolow.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        print('Found a frozen YOLOw model, defrost and use!')        
        tf.import_graph_def(graph_def)