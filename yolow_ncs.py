import os, sys, logging
from argparse import ArgumentParser

from openvino.inference_engine import IEPlugin, IENetwork

from predict import parse_yolo_region, intersection_over_union
from images import add_overlays_v2
from imager import Imager

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()

def get_yolow_ncs_arg_parser(arg_parser):
    arg_parser.add_argument('-p', '--model_name_path',
                            help='path to model file without the extension\
                                  (default to None and infered by model_type)',
                                 default=None)
    arg_parser.add_argument('-a', '--anchor_file',
                            help='path to the anchor file\
                                 (default to None and infered by model_type)',
                            default=None)
    arg_parser.add_argument('-c', '--num_classes',
                            help='number of classes to predict\
                                 (default to data/labels/coco.names)',
                            default=80,
                            type=int)    
    arg_parser.add_argument('-s', '--input_size',
                            help='an integer of the input size in accordance to the model\
                                 (default to 416)',
                            default=416,
                            type=int)
    return arg_parser

class YoloV3Params:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self, param, side):
        self.num = 3 if 'num' not in param else len(param['mask'].split(',')) if 'mask' in param else int(param['num'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 80 if 'classes' not in param else int(param['classes'])
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0,
                        373.0, 326.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]
        self.side = side
        if len(self.anchors) == 18: # yolov3
            if self.side == 13:
                self.anchor_offset = 2 * 6
            elif self.side == 26:
                self.anchor_offset = 2 * 3
            elif self.side == 52:
                self.anchor_offset = 2 * 0
            else:
                assert False, "Invalid output size. Only 13, 26 and 52 sizes are supported for output spatial dimensions"
        elif len(self.anchors) == 12: # yolov3-tiny
            if self.side == 13:
                self.anchor_offset = 2 * 3
            elif self.side == 26:
                self.anchor_offset = 2 * 0
            else:
                assert False, "Invalid output size. Only 13, 26 and 52 sizes are supported for output spatial dimensions"
        else:
            raise 'the number of anchors not supported'

    def log_params(self):
        params_to_print = {'classes': self.classes, 'num': self.num, 'coords': self.coords, 'anchors': self.anchors}
        [log.info("         {:8}: {}".format(param_name, param)) for param_name, param in params_to_print.items()]


class YolowNCS(object):

    def __init__(self, model_name_path, input_size, labels, num_requests=2):
        self.model = model_name_path + '.xml'
        self.weights = model_name_path + '.bin'
        self.labels = labels
        self.input_size = input_size
        self.imer = Imager(self.input_size, self.labels)
        if not os.path.exists(self.model) or not os.path.exists(self.weights):
            raise ValueError('model files {} does not exist.'.format(model_name_path))
        self.plugin=IEPlugin(device='MYRIAD')
        log.info('Loading network files:\n\t{}\n\t{}'.format(self.model, self.weights))
        self.net=IENetwork(model=self.model, weights=self.weights)
        log.info('Preparing inputs')
        self.input_blob=next(iter(self.net.inputs))
        self.net.batch_size=1
        log.info('Loading model to the plugin')
        self.current_request_id = 0
        self.next_request_id = 1
        self.num_requests = num_requests
        self.exec_net=self.plugin.load(network=self.net, num_requests=self.num_requests)

    def set_input(self, images):
        if type(images) == str:
            self.imer.imset_from_path(images)
        else:
            self.imer.imset(images)
        

    def predict(self, confidence_threshold=.5, iou_threshold=.4, async_mode=False):
        self.out_list = list()  
        in_frames = self.imer.ncs_preprocess()
        for in_frame, orig_frame in zip(in_frames, self.imer.ims):
            objects = list()
            origin_im_size = orig_frame.shape[:-1]
            input_dict = {self.input_blob: in_frame}
            request_handle = self.exec_net.requests[self.current_request_id]
            if async_mode:
                next_request_id = self.current_request_id + 1
                if next_request_id == self.num_requests:
                    next_request_id = 0
            else:
                next_request_id = self.current_request_id
            self.exec_net.start_async(request_id=next_request_id,
                                    inputs=input_dict)
            if async_mode:
                self.current_request_id = next_request_id
            request_handle.wait()
            pred_dict = request_handle.outputs
            for layer_name, out_blob in pred_dict.items():
                params = self.net.layers[layer_name].params
                layer_params = YoloV3Params(params, out_blob.shape[2])
                objects += parse_yolo_region(out_blob, in_frame.shape[2:],
                                                origin_im_size, layer_params,
                                                confidence_threshold)
            for i in range(len(objects)):
                if objects[i]['confidence'] == 0:
                    continue
                for j in range(i + 1, len(objects)):
                    if intersection_over_union(objects[i], objects[j]) > iou_threshold:
                        objects[j]['confidence'] = 0

            objects = [obj for obj in objects if obj['confidence'] >= confidence_threshold]
            self.out_list.append(objects)
        out_frames = self.imer.ncs_visualise_preds(self.out_list)
        return out_frames

    def get_output(self):
        try:
            return self.out_list
        except:
            raise ValueError('output does not exist, YolowNCS.predict() method must be called prior to this method.')

