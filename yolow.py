import tensorflow as tf
from yolov3 import *
from yolov3_tiny import *
from predict import predict
from load import *

class Yolow(object):
    
    sess = tf.Session()

    def  __init__(self, model_type='full', weight_path=None, input=None, is_training=False):
        self.model_type=model_type
        self.weight_path = weight_path
        try:
            self.defrost()
            self.input = tf.get_default_graph().get_tensor_by_name('import/input:0')
            self.output = tf.get_default_graph().get_tensor_by_name('import/detections/output:0')
        except:
            if not input:
                self.input = tf.placeholder(tf.float32, [None, 416, 416, 3], 'input')
            if self.model_type == 'full':
                # if self.weight_path != 'yolov3.weights':
                self.model = Yolov3(self.input, is_training)
            elif self.model_type == 'tiny':
                self.model = Yolov3Tiny(self.input, is_training)
            else:
                print('{} model type not supported! Yolow currently only supports \'full\' and \'tiny\' types'.format(self.model_type))
                exit()
            with tf.variable_scope('detections'):
                self.output = self.model.graph()
            self.loader = WeightLoader(tf.global_variables('detections'), self.model_type, weight_path)
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
        with tf.gfile.GFile('data/frozen_yolow_' + self.model_type + '.pb', 'wb') as f:
            f.write(graph_def.SerializeToString())

    def defrost(self):
        with tf.gfile.GFile('data/frozen_yolow_' + self.model_type + '.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        print('Found a frozen YOLOw model, defrost and use!')        
        tf.import_graph_def(graph_def)
        