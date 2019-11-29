from load import *
from yolov3 import *
from yolov3_tiny import *
from predict import predict
from argparse import ArgumentParser
from utils import *
from imager import Imager
import os
import tensorflow as tf

def get_yolow_arg_parser(arg_parser):
    arg_parser.add_argument('-m', '--model_type',
                            help='the type of model to use\
                                 (default to the full model)',
                            default='full',
                            type=str)
    arg_parser.add_argument('-f', '--model_file',
                            help='path to the weight/protobuf file\
                                  (default to None and infered by model_type)',
                                 default=None)
    arg_parser.add_argument('-a', '--anchor_file',
                            help='path to the anchor file\
                                 (default to None and infered by model_type)',
                            default=None)
    arg_parser.add_argument('-c', '--num_classes',
                            help='number of classes to predict\
                                 (default to 80)',
                            default=80,
                            type=int)    
    arg_parser.add_argument('-s', '--input_size',
                            help='an or a tuple of integer\
                                 (default to 416)',
                                 default=416,
                                 type=int)
    arg_parser.add_argument('-t', '--is_training',
                            help='Whether to train the model\
                                 (default to False)',
                            default=False,
                            type=bool)
    return arg_parser


class Yolow(object):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    # sess = tf.Session()
    default_models = load_default_models()
    model_types = list(default_models.keys())
    freeze_dir = 'data/pb/'
    tf_models = {model_types[0]: Yolov3,
                 model_types[1]: Yolov3Tiny}

    def  __init__(self, model_type,
                         model_file,
                         anchor_file,
                         num_classes,
                         input_size,
                         labels,
                         is_training=False):
        if model_type not in self.model_types:
            raise ValueError('model_type can only be either \'full\' or \'tiny\'.')
        elif not model_type:
            model_type = self.model_types[0]
        self.model_type = model_type

        if not model_file:
            model_file = './data/bin/{}'.format(self.default_models.get(model_type))
        elif not os.path.exists(model_file):
            raise ValueError('model file {} does not exist.'.format(model_file))
        self.model_file = model_file

        if '.pb' not in self.model_file:
            self.frozen_filename = '_'.join(['frozen', 
                                        os.path.basename(self.model_file).split('.')[0]])
            self.frozen_filename = self.freeze_dir + self.frozen_filename + '.pb'

        if not input_size:
            input_size = 416
        if type(input_size) is int:
            self.input_size = input_size, input_size
        else:
            self.input_size = input_size
        
        self.labels = labels
        self.imer = Imager(self.input_size, self.labels)
        
        if os.path.exists(self.frozen_filename):
            self.defrost()
            self.input = tf.get_default_graph().get_tensor_by_name('import/input:0')
            self.output = tf.get_default_graph().get_tensor_by_name('import/detections/output:0')
        else:
            if not anchor_file:
                anchor_file = 'data/anchors/' + self.model_type + '.txt'
            elif not os.path.exists(anchor_file):
                raise ValueError('{} anchor file does not exist.'.format(anchor_file))
            self.anchor_file = anchor_file
            self.num_classes = num_classes
            self.is_training = is_training
            self.input = tf.placeholder(tf.float32, 
                                        [None, self.input_size[0], self.input_size[1], 3], 
                                        'input')
            self.model = self.tf_models[self.model_type](self.input, 
                                                        self.num_classes, 
                                                        self.input_size, 
                                                        self.anchor_file, 
                                                        self.is_training)
            with tf.variable_scope('detections'):
                self.output = self.model.graph()
            self.loader = WeightLoader(tf.global_variables('detections'), self.model_file)
            # self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.loader.load_now())
            self.freeze()

    def set_input(self, images):
        if type(images) == str:
            self.imer.imset_from_path(images)
        else:
            self.imer.imset(images)


    def predict(self, confidence_theshold=.6, iou_threshold=.5):
        input_list = self.imer.preprocess()
        feed_dict = {self.input: input_list}
        batch_detections = self.sess.run(self.output, feed_dict)
        pred_list = predict(batch_detections, confidence_theshold, iou_threshold)
        return self.imer.visualise_preds(pred_list)

    def freeze(self):
        graph_def = tf.graph_util.convert_variables_to_constants(sess=self.sess,
                                                                 input_graph_def=tf.get_default_graph().as_graph_def(),
                                                                 output_node_names=['detections/output'])
        if not os.path.exists(self.freeze_dir):
            os.makedirs(self.freeze_dir)
        with tf.gfile.GFile(self.frozen_filename, 'wb') as f:
            f.write(graph_def.SerializeToString())


    def defrost(self):
        print('Found frozen model {}, defrost and use!'.format(self.frozen_filename))        
        with tf.gfile.GFile(self.frozen_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def)
        